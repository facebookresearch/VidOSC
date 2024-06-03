#
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import torch
from torch.utils.data.distributed import DistributedSampler
from dataset import HowToChangeFeatDataset, HowToChangeFeatCLIPLabelDataset


def construct_loader(args, split, dataset=None):
    assert split in ['train', 'val']
    if split == 'train':
        dataset = HowToChangeFeatCLIPLabelDataset(args)
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True) if args.gpus > 1 else None
        shuffle = False if args.gpus > 1 else True
    else:
        dataset = HowToChangeFeatDataset(args)
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=False) if args.gpus > 1 else None
        shuffle = False
    batch_size = args.batch_size if split == "train" else 1
    drop_last = True if split == "train" else False
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True,
        sampler=sampler,
        num_workers=args.num_workers)
    return loader
