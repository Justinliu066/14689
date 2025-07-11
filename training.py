import argparse
import datetime
import hashlib
import os
import shutil
import socket
import sys

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F

from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from .util import enumerateWithEstimate
from .dsets import Luna2dSegmentationDataset, TrainingLuna2dSegmentationDataset, getCt
from .logconf import logging
from .model import UNetWrapper, SegmentationAugmentation

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# Used for computeClassificationLoss and logMetrics to index into metrics_t/metrics_a
# METRICS_LABEL_NDX = 0
METRICS_LOSS_NDX = 1
# METRICS_FN_LOSS_NDX = 2
# METRICS_ALL_LOSS_NDX = 3

# METRICS_PTP_NDX = 4
# METRICS_PFN_NDX = 5
# METRICS_MFP_NDX = 6
METRICS_TP_NDX = 7
METRICS_FN_NDX = 8
METRICS_FP_NDX = 9

METRICS_SIZE = 10

class SegmentationTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=64,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=4,
            type=int,
        )
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=100,
            type=int,
        )

        parser.add_argument('--augmented',
            help="Augment the training data.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-flip',
            help="Augment the training data by randomly flipping the data left-right, up-down, and front-back.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-offset',
            help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-scale',
            help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-rotate',
            help="Augment the training data by randomly rotating the data around the head-foot axis.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-noise',
            help="Augment the training data by randomly adding noise to the data.",
            action='store_true',
            default=False,
        )

        parser.add_argument('--tb-prefix',
            default='p2ch13',
            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
        )

        parser.add_argument('comment',
            help="Comment suffix for Tensorboard run.",
            nargs='?',
            default='none',
        )
        parser.add_argument('--balanced', action='store_true', 
                      help='Enable class-balanced loss')
        parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
        parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
        parser.add_argument('--min-delta', type=float, default=0.001, help='minimum improvement for early stopping')
        
        self.scaler = torch.amp.GradScaler(device='cuda')
        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.totalTrainingSamples_count = 0
        self.trn_writer = None
        self.val_writer = None
        
        # 早停相关变量
        self.best_score = 0.0
        self.patience_counter = 0

        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.03
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.segmentation_model, self.augmentation_model = self.initModel()
        self.optimizer = self.initOptimizer()
        self.scheduler = self.initScheduler()


    def initModel(self):
        segmentation_model = UNetWrapper(
            in_channels=7,
            n_classes=1,
            depth=3,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
        )

        augmentation_model = SegmentationAugmentation(**self.augmentation_dict)

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                segmentation_model = nn.DataParallel(segmentation_model)
                augmentation_model = nn.DataParallel(augmentation_model)
            segmentation_model = segmentation_model.to(self.device)
            augmentation_model = augmentation_model.to(self.device)

        return segmentation_model, augmentation_model

    def initOptimizer(self):
        return Adam(
            self.segmentation_model.parameters(),
            lr=self.cli_args.lr,
            weight_decay=self.cli_args.weight_decay,
            betas=(0.9, 0.999)
        )
        # return SGD(self.segmentation_model.parameters(), lr=0.001, momentum=0.99)

    def initScheduler(self):
        # 使用 ReduceLROnPlateau 调度器，当验证指标不再改善时降低学习率
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # 监控指标越大越好
            factor=0.5,  # 学习率衰减因子
            patience=5,  # 5个epoch没有改善就降低学习率
            min_lr=1e-6
        )

    def initTrainDl(self):
        train_ds = TrainingLuna2dSegmentationDataset(
            val_stride=10,
            isValSet_bool=False,
            contextSlices_count=3,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl

    def initValDl(self):
        val_ds = Luna2dSegmentationDataset(
            val_stride=10,
            isValSet_bool=True,
            contextSlices_count=3,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '_trn_seg_' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '_val_seg_' + self.cli_args.comment)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        best_score = 0.0
        self.validation_cadence = 2  # 每2个epoch验证一次
        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            if epoch_ndx == 1 or epoch_ndx % self.validation_cadence == 0:
                # if validation is wanted
                valMetrics_t = self.doValidation(epoch_ndx, val_dl)
                score = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
                
                # 学习率调度
                self.scheduler.step(score)
                
                # 早停检查
                if score > best_score + self.cli_args.min_delta:
                    best_score = score
                    self.patience_counter = 0
                    self.saveModel('seg', epoch_ndx, True)  # 保存最佳模型
                else:
                    self.patience_counter += 1
                    self.saveModel('seg', epoch_ndx, False)
                
                # 早停
                if self.patience_counter >= self.cli_args.patience:
                    log.info(f"Early stopping triggered after {epoch_ndx} epochs")
                    break

                self.logImages(epoch_ndx, 'trn', train_dl)
                self.logImages(epoch_ndx, 'val', val_dl)

        self.trn_writer.close()
        self.val_writer.close()
    def doTraining(self, epoch_ndx, train_dl):
        trnMetrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)
        self.segmentation_model.train()
        train_dl.dataset.shuffleSamples()

        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(batch_ndx, batch_tup, train_dl.batch_size, trnMetrics_g)
            loss_var.backward()

            self.optimizer.step()

        self.totalTrainingSamples_count += trnMetrics_g.size(1)

        return trnMetrics_g.to('cpu')
    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            valMetrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)
            self.segmentation_model.eval()

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g,
                         classificationThreshold=0.5):
        input_t, label_t, series_list, _slice_ndx_list = batch_tup

        # 确保输入和标签都是浮点类型
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)
        
        # 确保label是float类型
        label_g = self._ensure_float(label_g)
            
        if self.segmentation_model.training and self.augmentation_dict:
            input_g, label_g = self.augmentation_model(input_g, label_g)
            # 确保数据增强后label仍是float类型
            label_g = self._ensure_float(label_g)

        prediction_g = self.segmentation_model(input_g)

        # 确保预测值在[0,1]范围内
        if prediction_g.max() > 1 or prediction_g.min() < 0:
            prediction_g = torch.clamp(prediction_g, 0, 1)

        # 组合损失函数
        diceLoss_g = self.diceLoss(prediction_g, label_g)
        bceLoss_g = self.bceLoss(prediction_g, label_g)
        focalLoss_g = self.focalLoss(prediction_g, label_g)
        
        # 增加FN惩罚
        fnLoss_g = self.diceLoss(prediction_g * label_g, label_g)
        
        # 调整损失权重组合，增加FN惩罚权重
        total_loss = (0.4 * diceLoss_g.mean() + 
                     0.2 * bceLoss_g.mean() + 
                     0.2 * focalLoss_g.mean() +
                     0.2 * fnLoss_g.mean())

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + input_t.size(0)

        with torch.no_grad():
            predictionBool_g = (prediction_g[:, 0:1]
                                > classificationThreshold).to(torch.float32)

            # 使用浮点操作计算指标
            tp = (predictionBool_g * label_g).sum(dim=[1,2,3])
            fn = ((1 - predictionBool_g) * label_g).sum(dim=[1,2,3])
            
            # 转为布尔值再计算
            label_bool = label_g > 0.5
            fp = (predictionBool_g * (~label_bool)).float().sum(dim=[1,2,3])

            metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = total_loss
            metrics_g[METRICS_TP_NDX, start_ndx:end_ndx] = tp
            metrics_g[METRICS_FN_NDX, start_ndx:end_ndx] = fn
            metrics_g[METRICS_FP_NDX, start_ndx:end_ndx] = fp

        return total_loss

    def _ensure_float(self, tensor):
        """确保张量是浮点类型"""
        if tensor.dtype != torch.float32:
            return tensor.float()
        return tensor

    def bceLoss(self, prediction_g, label_g):
        """Binary Cross Entropy Loss"""
        # 确保输入都是浮点类型
        label_g = self._ensure_float(label_g)
        prediction_g = self._ensure_float(prediction_g)
            
        # 确保值在[0,1]范围内
        prediction_g = torch.clamp(prediction_g, 1e-7, 1-1e-7)
        
        # 计算正负样本权重
        pos_weight = torch.ones_like(label_g) * 3.0  # 正样本权重为3
        
        # 使用带权重的BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(
            torch.log(prediction_g / (1 - prediction_g)),  # 转换为logits
            label_g,
            pos_weight=pos_weight,
            reduction='none'
        )
        
        return bce_loss.mean(dim=[1,2,3])

    def focalLoss(self, prediction_g, label_g, alpha=0.75, gamma=2.0):
        """Focal Loss for handling class imbalance"""
        # 确保输入都是浮点类型
        label_g = self._ensure_float(label_g)
        prediction_g = self._ensure_float(prediction_g)
            
        # 确保值在[0,1]范围内防止数值不稳定
        prediction_g = torch.clamp(prediction_g, 1e-7, 1-1e-7)
        
        bce_loss = F.binary_cross_entropy(prediction_g, label_g, reduction='none')
        pt = torch.exp(-bce_loss)
        # 正样本权重增加到0.75
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean(dim=[1,2,3])

    def diceLoss(self, prediction_g, label_g, epsilon=1):
        # 确保输入都是浮点类型
        label_g = self._ensure_float(label_g)
        prediction_g = self._ensure_float(prediction_g)
            
        diceLabel_g = label_g.sum(dim=[1,2,3])
        dicePrediction_g = prediction_g.sum(dim=[1,2,3])
        diceCorrect_g = (prediction_g * label_g).sum(dim=[1,2,3])

        diceRatio_g = (2 * diceCorrect_g + epsilon) \
            / (dicePrediction_g + diceLabel_g + epsilon)

        return 1 - diceRatio_g


    def logImages(self, epoch_ndx, mode_str, dl):
        self.segmentation_model.eval()

        images = sorted(dl.dataset.series_list)[:12]
        for series_ndx, series_uid in enumerate(images):
            ct = getCt(series_uid)

            for slice_ndx in range(6):
                ct_ndx = slice_ndx * (ct.hu_a.shape[0] - 1) // 5
                sample_tup = dl.dataset.getitem_fullSlice(series_uid, ct_ndx)

                ct_t, label_t, series_uid, ct_ndx = sample_tup

                input_g = ct_t.to(self.device).unsqueeze(0)
                label_g = pos_g = label_t.to(self.device).unsqueeze(0)
                
                # 确保标签是浮点类型
                if label_g.dtype != torch.float:
                    label_g = label_g.float()
                
                with torch.no_grad():
                    prediction_g = self.segmentation_model(input_g)[0]
                
                # 转为numpy前确保是CPU张量
                prediction_a = prediction_g.detach().cpu().numpy()[0] > 0.5
                label_a = label_g.cpu().detach().numpy()[0][0] > 0.5

                ct_t[:-1,:,:] /= 2000
                ct_t[:-1,:,:] += 0.5

                ctSlice_a = ct_t[dl.dataset.contextSlices_count].numpy()

                image_a = np.zeros((512, 512, 3), dtype=np.float32)
                image_a[:,:,:] = ctSlice_a.reshape((512,512,1))
                image_a[:,:,0] += prediction_a & (1 - label_a)
                image_a[:,:,0] += (1 - prediction_a) & label_a
                image_a[:,:,1] += ((1 - prediction_a) & label_a) * 0.5

                image_a[:,:,1] += prediction_a & label_a
                image_a *= 0.5
                image_a.clip(0, 1, image_a)

                writer = getattr(self, mode_str + '_writer')
                writer.add_image(
                    f'{mode_str}/{series_ndx}_prediction_{slice_ndx}',
                    image_a,
                    self.totalTrainingSamples_count,
                    dataformats='HWC',
                )

                if epoch_ndx == 1:
                    image_a = np.zeros((512, 512, 3), dtype=np.float32)
                    image_a[:,:,:] = ctSlice_a.reshape((512,512,1))
                    # image_a[:,:,0] += (1 - label_a) & lung_a # Red
                    image_a[:,:,1] += label_a  # Green
                    # image_a[:,:,2] += neg_a  # Blue

                    image_a *= 0.5
                    image_a[image_a < 0] = 0
                    image_a[image_a > 1] = 1
                    writer.add_image(
                        '{}/{}_label_{}'.format(
                            mode_str,
                            series_ndx,
                            slice_ndx,
                        ),
                        image_a,
                        self.totalTrainingSamples_count,
                        dataformats='HWC',
                    )
                # This flush prevents TB from getting confused about which
                # data item belongs where.
                writer.flush()

    def logMetrics(self, epoch_ndx, mode_str, metrics_t):
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        metrics_a = metrics_t.detach().numpy()
        sum_a = metrics_a.sum(axis=1)
        assert np.isfinite(metrics_a).all()

        allLabel_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_NDX].mean()

        metrics_dict['percent_all/tp'] = \
            sum_a[METRICS_TP_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fn'] = \
            sum_a[METRICS_FN_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fp'] = \
            sum_a[METRICS_FP_NDX] / (allLabel_count or 1) * 100


        precision = metrics_dict['pr/precision'] = sum_a[METRICS_TP_NDX] \
            / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX]) or 1)
        recall    = metrics_dict['pr/recall']    = sum_a[METRICS_TP_NDX] \
            / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]) or 1)

        metrics_dict['pr/f1_score'] = 2 * (precision * recall) \
            / ((precision + recall) or 1)

        log.info(("E{} {:8} "
                 + "{loss/all:.4f} loss, "
                 + "{pr/precision:.4f} precision, "
                 + "{pr/recall:.4f} recall, "
                 + "{pr/f1_score:.4f} f1 score"
                  ).format(
            epoch_ndx,
            mode_str,
            **metrics_dict,
        ))
        log.info(("E{} {:8} "
                  + "{loss/all:.4f} loss, "
                  + "{percent_all/tp:-5.1f}% tp, {percent_all/fn:-5.1f}% fn, {percent_all/fp:-9.1f}% fp"
        ).format(
            epoch_ndx,
            mode_str + '_all',
            **metrics_dict,
        ))

        self.initTensorboardWriters()
        writer = getattr(self, mode_str + '_writer')

        prefix_str = 'seg_'

        for key, value in metrics_dict.items():
            writer.add_scalar(prefix_str + key, value, self.totalTrainingSamples_count)

        writer.flush()

        score = metrics_dict['pr/recall']

        return score

    # def logModelMetrics(self, model):
    #     writer = getattr(self, 'trn_writer')
    #
    #     model = getattr(model, 'module', model)
    #
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             min_data = float(param.data.min())
    #             max_data = float(param.data.max())
    #             max_extent = max(abs(min_data), abs(max_data))
    #
    #             # bins = [x/50*max_extent for x in range(-50, 51)]
    #
    #             writer.add_histogram(
    #                 name.rsplit('.', 1)[-1] + '/' + name,
    #                 param.data.cpu().numpy(),
    #                 # metrics_a[METRICS_PRED_NDX, negHist_mask],
    #                 self.totalTrainingSamples_count,
    #                 # bins=bins,
    #             )
    #
    #             # print name, param.data

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            'data-unversioned',
            'part2',
            'models',
            self.cli_args.tb_prefix,
            f'{type_str}_{self.time_str}_{self.cli_args.comment}.pth'  # 修改后缀
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.segmentation_model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'input_channels': model.in_channels,  # 添加模型结构参数
            'n_classes': model.n_classes,
            'depth': model.depth,
        }

        torch.save(state, file_path)  # 仅保存必要信息

        if isBest:
            best_path = os.path.join(
                'data-unversioned', 'part2', 'models',
                self.cli_args.tb_prefix,
                f'{type_str}_{self.time_str}_{self.cli_args.comment}.best.pth'  # 最佳模型
            )
            shutil.copyfile(file_path, best_path)

            log.info("Saved model params to {}".format(best_path))

        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())


if __name__ == '__main__':
    SegmentationTrainingApp().main()
