import argparse
import sys

import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from .dsets import LunaDataset, getCtRawCandidate, getCt, getCandidateInfoList
from util.logconf import logging
# from .model import LunaModel

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)


class LunaPrepCacheApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=1024,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        # parser.add_argument('--scaled',
        #     help="Scale the CT chunks to square voxels.",
        #     default=False,
        #     action='store_true',
        # )

        self.cli_args = parser.parse_args(sys_argv)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        # 明确类型为LunaDataset
        self.prep_dl = DataLoader(
            LunaDataset(
                sortby_str='series_uid',
            ),
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
        )

        batch_iter = enumerateWithEstimate(
            self.prep_dl,
            "Stuffing cache",
            start_ndx=self.prep_dl.num_workers,
        )
        
        # 预先获取所有candidate信息，避免从dataset直接访问
        candidateInfo_list = getCandidateInfoList()
        
        width_irc = (32, 48, 48)
        
        for batch_ndx, batch_tup in batch_iter:
            # 实际缓存候选区域
            candidate_t, label_t, index_t, series_uid, center_irc = batch_tup
            
            # 通过访问每个批次的样本来触发memoize缓存
            for i, series_uid_i in enumerate(series_uid):
                # 直接使用series_uid触发缓存，不需要访问candidateInfo_list
                ct = getCt(series_uid_i)
                
                # 对于每个series_uid，找到对应的所有候选
                candidates = [c for c in candidateInfo_list if c.series_uid == series_uid_i]
                
                # 触发缓存
                for candidate in candidates:
                    getCtRawCandidate(
                        series_uid=series_uid_i,
                        center_xyz=candidate.center_xyz,
                        width_irc=width_irc
                    )
                    
            log.debug(f"Cached batch {batch_ndx}, {len(series_uid)} series")


if __name__ == '__main__':
    LunaPrepCacheApp().main()
