#
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
import argparse
import os
import json
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from task import FrameCls


def parse_args():
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--ann_dir", type=str, default='./data_files')
    parser.add_argument("--pseudolabel_dir", type=str, default="./videoclip_pseudolabel")
    parser.add_argument("--feat_dir", type=str, default='./data')
    parser.add_argument("--sc_list", type=str, nargs='+', default=['peeling'])
    # model args
    parser.add_argument("--transformer_heads", type=int, default=4)
    parser.add_argument("--transformer_layers", type=int, default=3)
    parser.add_argument("--transformer_dim", type=int, default=512)
    parser.add_argument("--transformer_dropout", type=float, default=0.1)
    # training args
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=0.0001)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--log_name", type=str, default="debug")
    parser.add_argument("--ckpt", type=str, default="")

    parser.add_argument("--use_gt_action", action="store_true")  # multitask use gt state transition idx
    parser.add_argument("--det", default=0, type=int, choices=[0, 1])

    return parser.parse_args()


def main():
    log_dir = os.path.join(args.log_dir, args.log_name)
    os.makedirs(log_dir, exist_ok=True)
    print('Logging to:', log_dir)
    with open(f'{log_dir}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    task = FrameCls(args)

    checkpoint_callbacks = []
    if hasattr(task, 'metric_name_list'):
        for name in task.metric_name_list:
            checkpoint_callbacks.append(
                ModelCheckpoint(
                    monitor=name,
                    filename='model-{epoch:02d}-{' + name + ':.4f}',
                    save_top_k=1,
                    mode='max',
                )
            )

    trainer = pl.Trainer(
        devices=args.gpus,
        accelerator="gpu",
        max_epochs=args.n_epochs,
        default_root_dir=log_dir,
        logger=TensorBoardLogger(save_dir=log_dir),
        callbacks=checkpoint_callbacks,
        num_sanity_val_steps=0,
    )

    if args.ckpt != "":
        task = task.load_from_checkpoint(checkpoint_path=args.ckpt, args=args)
        print(f'Evaluating {args.ckpt}')
        trainer.validate(task)
    else:
        trainer.fit(task)


if __name__ == '__main__':
    args = parse_args()
    main()
