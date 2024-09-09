#
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import pytorch_lightning as pl
from collections import defaultdict
import lookforthechange

from dataset_changeit import ChangeItFeatCLIPLabelDataset, ChangeItFeatDataset
from loader import construct_loader
from model import FeatTimeTransformerDualHead
from data_scripts.evaluator import StateMetric, ActionMetric


class FrameCls(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_ds = ChangeItFeatCLIPLabelDataset(args, "train")
        self.val_ds = ChangeItFeatDataset(args, "val")

        args.input_dim, self.n_classes, self.classid2name, args.state_classes, args.action_classes, args.object_classes, \
        self.state_id_mapping, self.action_id_mapping, self.action_to_state, args.vocab_size = self.train_ds.get_state_action_mapping()
        self.action_bg_idx = self.train_ds.action_bg_idx

        self.model = FeatTimeTransformerDualHead(args)

        self.state_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        self.action_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

        self.state_metric = {i: StateMetric() for i in range(self.n_classes)}
        self.action_metric = {i: ActionMetric() for i in range(self.n_classes)}
        self.metric_name_list = ['val_avg_prec_1', 'val_avg_action_prec_1']
        self.ckpt_save_format = 'model-{epoch:02d}-{' + self.metric_name_list[0] + ':.4f}-{' + self.metric_name_list[1] + ':.4f}'
        
        if self.args.novel_obj:
            self.test_ds = ChangeItFeatDataset(args, "test")
            _, self.n_classes_test, self.classid2name_test, *_, self.action_to_state_test, _ = self.test_ds.get_state_action_mapping()
            self.state_metric_test = {i: StateMetric() for i in range(self.n_classes_test)}
            self.action_metric_test = {i: ActionMetric() for i in range(self.n_classes_test)}
            self.test_to_train_action_mapping = {
                0: 0, 1: 1, 2: 8, 3: 4, 4: 3, 5: 2, 6: 6, 7: 5, 8: 7
            }


    def training_step(self, batch, batch_idx):
        input, state_pl, action_pl, video_level_score = batch   # text_embed 16, 768
        bs = input.shape[0]
        pred = self.model(input)  # [bs * ts, n_classes]

        state_loss_unweighted = self.state_loss(pred['state'], state_pl.view(-1))
        state_loss = (state_loss_unweighted.reshape(bs, -1) * video_level_score[:, None]).sum() / (state_pl != -1).sum()
        action_loss_unweighted = self.action_loss(pred['action'], action_pl.view(-1))
        action_loss = (action_loss_unweighted.reshape(bs, -1) * video_level_score[:, None]).sum() / (action_pl != -1).sum()
        loss = state_loss + action_loss

        self.log("train_state_loss", state_loss, on_epoch=True)
        self.log("train_action_loss", action_loss, on_epoch=True)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def model_inference(self, input):
        pred = self.model(input)
        st_probs = torch.softmax(pred['state'].unsqueeze(0), -1)  # (1, 900, 2*n_classes+1)
        ac_probs = torch.softmax(pred['action'].unsqueeze(0), -1)
        return st_probs, ac_probs

    def validation_step(self, batch, batch_idx):
        _, class_, input, annot, _ = batch
        class_ = class_.item()
        annot = annot.squeeze(0)
        st_probs, ac_probs = self.model_inference(input)
        st_probs = st_probs[0]
        ac_probs = ac_probs[0]
        if self.args.use_gt_action:
            action_idx = self.action_id_mapping[class_]
            s0_idx, s1_idx = self.state_id_mapping[2 * class_], self.state_id_mapping[2 * class_ + 1]
        else:
            sum_ac_probs = ac_probs.sum(dim=0)
            sum_ac_probs[self.action_bg_idx] = torch.tensor(float('-inf'))
            action_idx = sum_ac_probs.argmax().item()
            s0_idx, s1_idx = self.action_to_state[action_idx]

        st_probs = torch.stack((st_probs[:, s0_idx], st_probs[:, s1_idx]), dim=-1)
        ac_probs = ac_probs[:, action_idx]

        if len(annot) > len(st_probs):
            annot = annot[:len(st_probs)]
        if len(st_probs) > len(annot):
            st_probs = st_probs[:len(annot), :]
            ac_probs = ac_probs[:len(annot)]

        if self.args.ordering:
            indices = lookforthechange.optimal_state_change_indices(st_probs.unsqueeze(0), ac_probs.unsqueeze(0).unsqueeze(-1),
                        lens=torch.tensor([st_probs.shape[0]], dtype=torch.int32, device=st_probs.device))[0]
            state_idx = indices[0:2]
            action_idx = indices[-1]
        else:
            state_idx = torch.argmax(st_probs, dim=0)
            action_idx = torch.argmax(ac_probs)

        self.state_metric[class_].update(state_idx, annot)
        self.action_metric[class_].update(action_idx, annot)

    def on_validation_epoch_end(self):
        total_metrics = defaultdict(float)
        for i in range(self.n_classes):
            result = self.state_metric[i].compute()
            name = self.classid2name[i]
            for key, value in result.items():
                if key == 'prec_1':
                    self.log(f'val_{key}_{name}', value, on_epoch=True, prog_bar=False, sync_dist=True)
                total_metrics[key] += value
            self.state_metric[i].reset()

            action_prec_1 = self.action_metric[i].compute()
            self.log(f'val_action_prec_1_{name}', action_prec_1, on_epoch=True, prog_bar=False, sync_dist=True)
            total_metrics['action_prec_1'] += action_prec_1
            self.action_metric[i].reset()

        for key, total_value in total_metrics.items():
            avg_value = total_value / self.n_classes
            self.log('val_avg_' + key, avg_value, on_epoch=True, prog_bar=True)


    def test_step(self, batch, batch_idx):
        _, class_, input, annot, _ = batch
        class_ = class_.item()
        annot = annot.squeeze(0)
        st_probs, ac_probs = self.model_inference(input)
        st_probs = st_probs[0]
        ac_probs = ac_probs[0]
        if self.args.use_gt_action:
            action_idx = self.test_to_train_action_mapping[self.action_id_mapping[class_]]
        else:
            sum_ac_probs = ac_probs.sum(dim=0)
            sum_ac_probs[self.action_bg_idx] = torch.tensor(float('-inf'))
            action_idx = sum_ac_probs.argmax().item()

        s0_idx, s1_idx = self.action_to_state_test[action_idx]
        st_probs = torch.stack((st_probs[:, s0_idx], st_probs[:, s1_idx]), dim=-1)
        ac_probs = ac_probs[:, action_idx]

        if len(annot) > len(st_probs):
            annot = annot[:len(st_probs)]
        if len(st_probs) > len(annot):
            st_probs = st_probs[:len(annot), :]
            ac_probs = ac_probs[:len(annot)]

        if self.args.ordering:
            indices = lookforthechange.optimal_state_change_indices(st_probs.unsqueeze(0), ac_probs.unsqueeze(0).unsqueeze(-1),
                    lens=torch.tensor([st_probs.shape[0]], dtype=torch.int32, device=st_probs.device))[0]
            state_idx = indices[0:2]
            action_idx = indices[-1]
        else:
            state_idx = torch.argmax(st_probs, dim=0)
            action_idx = torch.argmax(ac_probs)

        self.state_metric_test[class_].update(state_idx, annot)
        self.action_metric_test[class_].update(action_idx, annot)

    def on_test_epoch_end(self):
        total_metrics = defaultdict(float)
        for i in range(self.n_classes_test):
            result = self.state_metric_test[i].compute()
            name = self.classid2name_test[i]
            for key, value in result.items():
                if key == 'prec_1':
                    self.log('test_' + key + '_' + name, value, on_epoch=True, prog_bar=False)
                total_metrics[key] += value
            self.state_metric_test[i].reset()

            action_prec_1 = self.action_metric_test[i].compute()
            self.log('test_action_prec_1_' + name, action_prec_1, on_epoch=True, prog_bar=False)
            total_metrics['action_prec_1'] += action_prec_1
            self.action_metric_test[i].reset()

        for key, total_value in total_metrics.items():
            avg_value = total_value / self.n_classes_test
            self.log('test_avg_' + key, avg_value, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        return optimizer

    def train_dataloader(self):
        return construct_loader(self.args, "train", self.train_ds)

    def val_dataloader(self):
        return construct_loader(self.args, "val", self.val_ds)

    def test_dataloader(self):
        return construct_loader(self.args, "test", self.test_ds)