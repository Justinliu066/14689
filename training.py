import argparse
import datetime
import hashlib
import os
import shutil
import sys

import numpy as np
from matplotlib import pyplot

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

import p2ch14.dsets
import p2ch14.model

from .util import enumerateWithEstimate
from .logconf import logging


log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# Used for computeBatchLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX=0
METRICS_PRED_NDX=1
METRICS_PRED_P_NDX=2
METRICS_LOSS_NDX=3
METRICS_SIZE = 4

class ClassificationTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=24,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=0,  # 默认值改为0，避免Windows上的多进程问题
            type=int,
        )
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=1,
            type=int,
        )
        parser.add_argument('--dataset',
            help="What to dataset to feed the model.",
            action='store',
            default='LunaDataset',
        )
        parser.add_argument('--model',
            help="What to model class name to use.",
            action='store',
            default='LunaModel',
        )
        parser.add_argument('--malignant',
            help="Train the model to classify nodules as benign or malignant.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--finetune',
            help="Start finetuning from this model.",
            default='',
        )
        parser.add_argument('--finetune-depth',
            help="Number of blocks (counted from the head) to include in finetuning",
            type=int,
            default=1,
        )
        parser.add_argument('--tb-prefix',
            default='p2ch14',
            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
        )
        parser.add_argument('--lr',
            help="Learning rate",
            type=float,
            default=0.001,
        )
        parser.add_argument('--lr-scheduler',
            help="Learning rate scheduler (cosine or plateau)",
            choices=['cosine', 'plateau', 'none'],
            default='none',
        )
        parser.add_argument('--optimizer',
            help="Optimizer to use (sgd, adam, or adamw)",
            choices=['sgd', 'adam', 'adamw'],
            default='sgd',
        )
        parser.add_argument('--weight-decay',
            help="Weight decay factor",
            type=float,
            default=1e-4,
        )
        parser.add_argument('--grad-clip',
            help="Gradient clipping value",
            type=float,
            default=0.0,
        )
        parser.add_argument('--early-stopping',
            help="Enable early stopping",
            action='store_true',
            default=False,
        )
        parser.add_argument('--patience',
            help="Patience for early stopping",
            type=int,
            default=5,
        )
        parser.add_argument('--mixed-precision',
            help="Use mixed precision training",
            action='store_true',
            default=False,
        )
        parser.add_argument('comment',
            help="Comment suffix for Tensorboard run.",
            nargs='?',
            default='dlwpt',
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.augmentation_dict = {}
        if True:
        # if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        # if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.1
        # if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        # if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        # if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # 简化混合精度训练 - 完全禁用，避免API兼容性问题
        self.use_amp = False
        self.scaler = None
        # 禁用混合精度训练，即使用户指定了这个选项
        if self.cli_args.mixed_precision and self.use_cuda:
            log.warning("混合精度训练已禁用，以避免版本兼容问题")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()
        # 初始化学习率调度器
        self.scheduler = self.initScheduler()
        
        # 早停相关参数
        self.best_score = 0.0
        self.patience_count = 0
        self.early_stopping_triggered = False

    def initModel(self):
        model_cls = getattr(p2ch14.model, self.cli_args.model)
        model = model_cls()

        if self.cli_args.finetune:
            d = torch.load(self.cli_args.finetune, map_location='cpu')
            model_blocks = [
                n for n, subm in model.named_children()
                if len(list(subm.parameters())) > 0
            ]
            finetune_blocks = model_blocks[-self.cli_args.finetune_depth:]
            log.info(f"finetuning from {self.cli_args.finetune}, blocks {' '.join(finetune_blocks)}")
            model.load_state_dict(
                {
                    k: v for k,v in d['model_state'].items()
                    if k.split('.')[0] not in model_blocks[-1]
                },
                strict=False,
            )
            for n, p in model.named_parameters():
                if n.split('.')[0] not in finetune_blocks:
                    p.requires_grad_(False)
        if self.use_cuda:
            try:
                log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
                if torch.cuda.device_count() > 1:
                    model = nn.DataParallel(model)
                model = model.to(self.device)
            except RuntimeError as e:
                log.error(f"CUDA错误: {e}")
                log.info("回退到CPU模式")
                self.use_cuda = False
                self.device = torch.device("cpu")
                model = model.to(self.device)
        return model

    def initOptimizer(self):
        # 使用命令行指定的优化器和学习率
        lr = self.cli_args.lr if not self.cli_args.finetune else self.cli_args.lr * 0.5
        weight_decay = self.cli_args.weight_decay
        
        if self.cli_args.optimizer == 'sgd':
            return SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif self.cli_args.optimizer == 'adam':
            return Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif self.cli_args.optimizer == 'adamw':
            return AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            return SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def initScheduler(self):
        # 为避免版本兼容性问题，始终返回None
        log.warning("学习率调度器已被禁用，以避免版本兼容性问题")
        return None
            
    def initTrainDl(self):
        ds_cls = getattr(p2ch14.dsets, self.cli_args.dataset)

        train_ds = ds_cls(
            val_stride=10,
            isValSet_bool=False,
            ratio_int=1,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
            persistent_workers=False,  # Windows上避免使用持久化工作进程
        )

        return train_dl

    def initValDl(self):
        ds_cls = getattr(p2ch14.dsets, self.cli_args.dataset)

        val_ds = ds_cls(
            val_stride=10,
            isValSet_bool=True,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
            persistent_workers=False,  # Windows上避免使用持久化工作进程
        )

        return val_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix,
                                   self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn_cls-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + self.cli_args.comment)


    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        best_score = 0.0
        validation_cadence = 5 if not self.cli_args.finetune else 1
        
        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            if self.early_stopping_triggered:
                log.info("Early stopping triggered. Stopping training.")
                break

            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            try:
                trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
                self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)
    
                if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
                    valMetrics_t = self.doValidation(epoch_ndx, val_dl)
                    score = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
                    
                    # 早停检查
                    if score > best_score:
                        best_score = score
                        self.saveModel('cls', epoch_ndx, True)
                        self.patience_count = 0
                    else:
                        self.patience_count += 1
                        if self.cli_args.early_stopping and self.patience_count >= self.cli_args.patience:
                            self.early_stopping_triggered = True
            except Exception as e:
                log.error(f"训练过程中出现错误: {e}")
                import traceback
                log.error(traceback.format_exc())
                if self.cli_args.num_workers > 0:
                    log.info("尝试减少工作进程数量，重新训练")
                    self.cli_args.num_workers = 0
                    # 重新初始化数据加载器
                    train_dl = self.initTrainDl()
                    val_dl = self.initValDl()
                    continue
                else:
                    # 如果已经是0个工作进程还出错，则退出
                    log.error("无法继续训练，退出")
                    break

        # 确保检查trn_writer和val_writer存在
        if self.trn_writer is not None:
            self.trn_writer.close()
        if self.val_writer is not None:
            self.val_writer.close()


    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        train_dl.dataset.shuffleSamples()
        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            # 标准训练代码，不使用混合精度
            loss_var = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g,
                augment=True
            )

            loss_var.backward()
            
            # 梯度裁剪
            if self.cli_args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cli_args.grad_clip
                )
            
            self.optimizer.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics_g.to('cpu')


    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(
                    batch_ndx,
                    batch_tup,
                    val_dl.batch_size,
                    valMetrics_g,
                    augment=False
                )

        return valMetrics_g.to('cpu')



    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g,
                         augment=True):
        input_t, label_t, index_t, _series_list, _center_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)
        index_g = index_t.to(self.device, non_blocking=True)


        if augment:
            input_g = p2ch14.model.augment3d(input_g)

        logits_g, probability_g = self.model(input_g)

        loss_g = nn.functional.cross_entropy(logits_g, label_g[:, 1],
                                             reduction="none")
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        _, predLabel_g = torch.max(probability_g, dim=1, keepdim=False,
                                   out=None)

        # log.debug(index_g)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = index_g
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = predLabel_g
        # metrics_g[METRICS_PRED_N_NDX, start_ndx:end_ndx] = probability_g[:,0]
        metrics_g[METRICS_PRED_P_NDX, start_ndx:end_ndx] = probability_g[:,1]
        # metrics_g[METRICS_PRED_M_NDX, start_ndx:end_ndx] = probability_g[:,2]
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g

        return loss_g.mean()


    def logMetrics(
            self,
            epoch_ndx,
            mode_str,
            metrics_t,
            classificationThreshold=0.5,
    ):
        self.initTensorboardWriters()
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        if self.cli_args.dataset == 'MalignantLunaDataset':
            pos = 'mal'
            neg = 'ben'
        else:
            pos = 'pos'
            neg = 'neg'


        negLabel_mask = metrics_t[METRICS_LABEL_NDX] == 0
        negPred_mask = metrics_t[METRICS_PRED_NDX] == 0

        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        # benLabel_mask = metrics_t[METRICS_LABEL_NDX] == 1
        # benPred_mask = metrics_t[METRICS_PRED_NDX] == 1
        #
        # malLabel_mask = metrics_t[METRICS_LABEL_NDX] == 2
        # malPred_mask = metrics_t[METRICS_PRED_NDX] == 2

        # benLabel_mask = ~malLabel_mask & posLabel_mask
        # benPred_mask = ~malPred_mask & posLabel_mask

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())
        # ben_count = int(benLabel_mask.sum())
        # mal_count = int(malLabel_mask.sum())

        neg_correct = int((negLabel_mask & negPred_mask).sum())
        pos_correct = int((posLabel_mask & posPred_mask).sum())
        # ben_correct = int((benLabel_mask & benPred_mask).sum())
        # mal_correct = int((malLabel_mask & malPred_mask).sum())

        trueNeg_count = neg_correct
        truePos_count = pos_correct

        falsePos_count = neg_count - neg_correct
        falseNeg_count = pos_count - pos_correct

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()
        # metrics_dict['loss/ben'] = metrics_t[METRICS_LOSS_NDX, benLabel_mask].mean()
        # metrics_dict['loss/mal'] = metrics_t[METRICS_LOSS_NDX, malLabel_mask].mean()

        metrics_dict['correct/all'] = (pos_correct + neg_correct) / metrics_t.shape[1] * 100
        metrics_dict['correct/neg'] = (neg_correct) / neg_count * 100
        metrics_dict['correct/pos'] = (pos_correct) / pos_count * 100
        # metrics_dict['correct/ben'] = (ben_correct) / ben_count * 100
        # metrics_dict['correct/mal'] = (mal_correct) / mal_count * 100

        precision = metrics_dict['pr/precision'] = \
            truePos_count / np.float64(truePos_count + falsePos_count)
        recall    = metrics_dict['pr/recall'] = \
            truePos_count / np.float64(truePos_count + falseNeg_count)

        metrics_dict['pr/f1_score'] = \
            2 * (precision * recall) / (precision + recall)

        # 修复linspace参数，根据版本选择合适的调用方式
        try:
            # 尝试使用steps参数(新版PyTorch)
            threshold = torch.linspace(1, 0, steps=100)
        except TypeError:
            # 如果出错，尝试使用旧版本的参数格式
            threshold = torch.linspace(1, 0, 100)
            
        tpr = (metrics_t[None, METRICS_PRED_P_NDX, posLabel_mask] >= threshold[:, None]).sum(1).float() / pos_count
        fpr = (metrics_t[None, METRICS_PRED_P_NDX, negLabel_mask] >= threshold[:, None]).sum(1).float() / neg_count
        fp_diff = fpr[1:]-fpr[:-1]
        tp_avg  = (tpr[1:]+tpr[:-1])/2
        auc = (fp_diff * tp_avg).sum()
        metrics_dict['auc'] = auc

        log.info(
            ("E{} {:8} {loss/all:.4f} loss, "
                 + "{correct/all:-5.1f}% correct, "
                 + "{pr/precision:.4f} precision, "
                 + "{pr/recall:.4f} recall, "
                 + "{pr/f1_score:.4f} f1 score, "
                 + "{auc:.4f} auc"
            ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/neg:.4f} loss, "
                 + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_' + neg,
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/pos:.4f} loss, "
                 + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_' + pos,
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )
        # log.info(
        #     ("E{} {:8} {loss/ben:.4f} loss, "
        #          + "{correct/ben:-5.1f}% correct ({ben_correct:} of {ben_count:})"
        #     ).format(
        #         epoch_ndx,
        #         mode_str + '_ben',
        #         ben_correct=ben_correct,
        #         ben_count=ben_count,
        #         **metrics_dict,
        #     )
        # )
        # log.info(
        #     ("E{} {:8} {loss/mal:.4f} loss, "
        #          + "{correct/mal:-5.1f}% correct ({mal_correct:} of {mal_count:})"
        #     ).format(
        #         epoch_ndx,
        #         mode_str + '_mal',
        #         mal_correct=mal_correct,
        #         mal_count=mal_count,
        #         **metrics_dict,
        #     )
        # )
        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            key = key.replace('pos', pos)
            key = key.replace('neg', neg)
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        # 添加学习率到TensorBoard
        writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.totalTrainingSamples_count)
        
        fig = pyplot.figure()
        pyplot.plot(fpr, tpr)
        writer.add_figure('roc', fig, self.totalTrainingSamples_count)

        writer.add_scalar('auc', auc, self.totalTrainingSamples_count)
# # tag::logMetrics_writer_prcurve[]
#        writer.add_pr_curve(
#            'pr',
#            metrics_t[METRICS_LABEL_NDX],
#            metrics_t[METRICS_PRED_P_NDX],
#            self.totalTrainingSamples_count,
#        )
# # end::logMetrics_writer_prcurve[]

        bins = np.linspace(0, 1)

        writer.add_histogram(
            'label_neg',
            metrics_t[METRICS_PRED_P_NDX, negLabel_mask],
            self.totalTrainingSamples_count,
            bins=bins
        )
        writer.add_histogram(
            'label_pos',
            metrics_t[METRICS_PRED_P_NDX, posLabel_mask],
            self.totalTrainingSamples_count,
            bins=bins
        )

        if not self.cli_args.malignant:
            score = metrics_dict['pr/f1_score']
        else:
            score = metrics_dict['auc']

        return score

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            'data-unversioned',
            'part2',
            'models',
            self.cli_args.tb_prefix,
            f'{type_str}_{self.time_str}_{self.cli_args.comment}.pth'  # 修改后缀
        )

        model = self.model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'model_state': model.state_dict(),
            'model_class': model.__class__.__name__,
            'input_shape': (1, 32, 48, 48),  # 示例输入尺寸
            'model_params': {                # 模型结构参数
                'in_channels': model.tail_batchnorm.num_features,
                'conv_channels': model.block1.conv1.out_channels
            }
        }

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)
        torch.save(state, file_path)

        if isBest:
            best_path = os.path.join(
                'data-unversioned',
                'part2',
                'models',
                self.cli_args.tb_prefix,
                f'{type_str}_{self.time_str}_{self.cli_args.comment}.best.pth'
            )
            shutil.copyfile(file_path, best_path)
            
            log.debug("Saved model params to {}".format(best_path))
            
            # 保存完整训练状态
            state_path = os.path.join(
                'data-unversioned',
                'part2',
                'models',
                self.cli_args.tb_prefix,
                '{}_{}_{}.{}.state'.format(
                    type_str,
                    self.time_str,
                    self.cli_args.comment,
                    'best',
                )
            )
            state = {
                'model_state': model.state_dict(),
                'model_name': type(model).__name__,
                'optimizer_state': self.optimizer.state_dict(),
                'optimizer_name': type(self.optimizer).__name__,
                'epoch': epoch_ndx,
                'totalTrainingSamples_count': self.totalTrainingSamples_count,
            }
            torch.save(state, state_path)
            
            log.debug("Saved training state to {}".format(state_path))
            
            with open(state_path, 'rb') as f:
                log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())

    def logModelMetrics(self, model):
        writer = getattr(self, 'trn_writer')
    
        model = getattr(model, 'module', model)
    
        for name, param in model.named_parameters():
            if param.requires_grad:
                min_data = float(param.data.min())
                max_data = float(param.data.max())
                max_extent = max(abs(min_data), abs(max_data))
    
                # bins = [x/50*max_extent for x in range(-50, 51)]
    
                try:
                    writer.add_histogram(
                        name.rsplit('.', 1)[-1] + '/' + name,
                        param.data.cpu().numpy(),
                        # metrics_a[METRICS_PRED_NDX, negHist_mask],
                        self.totalTrainingSamples_count,
                        # bins=bins,
                    )
                except Exception as e:
                    log.error([min_data, max_data])
                    raise


if __name__ == '__main__':
    ClassificationTrainingApp().main()
