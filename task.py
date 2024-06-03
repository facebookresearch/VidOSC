#
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import lookforthechange
from torchmetrics.classification import MulticlassPrecision, MulticlassF1Score

from dataset import build_vocab
from loader import construct_loader
from model import FeatTimeTransformer
from data_scripts.evaluator import StatePrec1


class FrameCls(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.infer_ordering = False  # use causual ordering constraint during inference
        self.vocab, self.sc_list, _ = build_vocab(args)
        self.category_num = len(self.vocab) - 1
        args.vocab_size = 3 * self.category_num + 1
        args.input_dim = 768 * (1 + self.args.det)
        self.model = FeatTimeTransformer(args)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.eval_setting = ['known', 'novel', 'all']
        self.state_prec1 = {sc: {key: StatePrec1() for key in self.eval_setting} for sc in self.sc_list}
        self.state_prec = MulticlassPrecision(num_classes=4, average="none")
        self.f1_score = MulticlassF1Score(num_classes=4, average="none")
        self.metric_name_list = ['avg_f1_known', 'avg_f1_novel', 'avg_prec_known', 'avg_prec_novel'] if len(
            self.sc_list) > 1 else [f'{self.sc_list[0]}_avg_f1']

    def training_step(self, batch, batch_idx):
        feat, pl = batch
        pred = self.model(feat)
        loss = self.loss(pred, pl.view(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def infer_state_idx(self, prob):
        pred_idx = torch.argmax(prob, dim=0).cpu().numpy()
        if self.infer_ordering:
            st_probs = torch.stack((prob[:, 1], prob[:, 3]), dim=-1).unsqueeze(0)
            ac_probs = prob[:, 2].unsqueeze(0).unsqueeze(2)
            # print('before', pred_idx)
            s0_idx, s2_idx, s1_idx = lookforthechange.optimal_state_change_indices(st_probs, ac_probs,
                   lens=torch.tensor([st_probs.shape[1]], dtype=torch.int32, device=st_probs.device))[0].cpu().numpy()
            pred_idx = np.array([s0_idx, s1_idx, s2_idx])
            # print('after', pred_idx)
        else:
            pred_idx = pred_idx[1:]
        return pred_idx

    def validation_step(self, batch, batch_idx):
        feat, label, osc, is_novel = batch
        osc = osc[0]
        sc_name = osc.split('_')[0]

        name = 'novel' if is_novel.item() else 'known'
        pred = self.model(feat)
        prob = torch.softmax(pred, dim=-1)
        category_pred = prob[:, 1:].reshape(-1, self.category_num, 3).sum(dim=0).sum(dim=-1)
        inferred_catgeory_id = category_pred.argmax().item() + 1
        key = sc_name
        gt_category_id = self.vocab[key]

        self.log('category_acc', inferred_catgeory_id == gt_category_id, on_step=False, on_epoch=True)
        category_id = gt_category_id if self.args.use_gt_action else inferred_catgeory_id
        st_prob = prob[:, [0, 3 * category_id - 2, 3 * category_id - 1, 3 * category_id]]

        pred_idx = self.infer_state_idx(st_prob)
        prec = self.state_prec(st_prob, label.view(-1))
        f1 = self.f1_score(st_prob, label.view(-1))
        self.state_prec1[sc_name][name].update(pred_idx, label.view(-1))
        self.state_prec1[sc_name]['all'].update(pred_idx, label.view(-1))
        unique_labels = torch.unique(label).cpu().numpy().astype(int)
        unique_labels = unique_labels[unique_labels > 0]
        avg_prec = prec[unique_labels].mean()
        avg_f1 = f1[unique_labels].mean()

        self.log(f'{sc_name}_avg_prec_{name}', avg_prec, on_step=False, on_epoch=True)
        self.log(f'{sc_name}_avg_prec', avg_prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'{sc_name}_avg_f1_{name}', avg_f1, on_step=False, on_epoch=True)
        self.log(f'{sc_name}_avg_f1', avg_f1, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        for i, sc_name in enumerate(self.sc_list):
            for key in self.eval_setting:
                val_prec1 = self.state_prec1[sc_name][key].compute()
                self.log(f'{sc_name}_avg_prec1_{key}', val_prec1['avg'], on_step=False, on_epoch=True, prog_bar=True)
                self.state_prec1[sc_name][key].reset()

        if len(self.sc_list) > 1:
            avg_result = np.zeros((len(self.sc_list), 6))
            value_name = ['avg_f1_known', 'avg_f1_novel', 'avg_prec_known', 'avg_prec_novel', 'avg_prec1_known',
                          'avg_prec1_novel']
            for i, sc_name in enumerate(self.sc_list):
                value_list = [self.trainer.callback_metrics.get(f'{sc_name}_{v}').item() for v in value_name]
                avg_result[i] = value_list
            avg_result = avg_result.mean(axis=0)
            for i, v in enumerate(value_name):
                self.log(f'{v}', avg_result[i], on_step=False, on_epoch=True, prog_bar=True)

        val_name = [f'{self.sc_list[0]}_avg_f1_known', f'{self.sc_list[0]}_avg_f1_novel',
                    f'{self.sc_list[0]}_avg_prec_known', f'{self.sc_list[0]}_avg_prec_novel',
                    f'{self.sc_list[0]}_avg_prec1_known', f'{self.sc_list[0]}_avg_prec1_novel']
        value_list = [round(self.trainer.callback_metrics.get(v).item() * 100, 2) for v in val_name]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        return optimizer

    def train_dataloader(self):
        return construct_loader(self.args, "train")

    def val_dataloader(self):
        return construct_loader(self.args, "val")
