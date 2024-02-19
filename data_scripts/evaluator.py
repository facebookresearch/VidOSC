#
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import pandas as pd
import numpy as np
import torch
from torchmetrics import Metric
from torchmetrics.classification import MulticlassPrecision, MulticlassF1Score
from data_scripts.read_ann import derive_label


class StatePrec1(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("prec1", default=torch.tensor([0.0, 0.0, 0.0, 0.0]), dist_reduce_fx="sum")
        self.add_state("cnt", default=torch.tensor([0, 0, 0, 0]), dist_reduce_fx="sum")

    def update(self, idx, gt):
        unique_labels = torch.unique(gt).cpu().numpy().astype(int)
        unique_labels = unique_labels[unique_labels > 0]
        correct_cnt = 0
        for i, label in enumerate(unique_labels):
            state = label - 1
            is_correct = (gt[idx[state]].item() == label)
            self.prec1[state] += is_correct
            correct_cnt += is_correct
            self.cnt[state] += 1
        self.prec1[-1] += (correct_cnt / len(unique_labels))
        self.cnt[-1] += 1

    def compute(self):
        prec1 = self.prec1 / self.cnt
        return {
            "s0": prec1[0].item(),
            "s1": prec1[1].item(),
            "s2": prec1[2].item(),
            "avg": prec1[3].item()
        }


class EvalClip:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df['state_transition'] = self.df['osc'].apply(lambda x: x.split('_')[0])
        self.sc_list = self.df['state_transition'].unique()
        self.fps = 1
        self.eval_setting = ['all', 'known', 'novel']
        self.state_prec1 = {key: StatePrec1() for key in self.eval_setting}
        self.state_prec = MulticlassPrecision(num_classes=4, average="none")
        self.f1_score = MulticlassF1Score(num_classes=4, average="none")
        self.prec_list = {key: {'s0': [], 's1': [], 's2': [], 'avg': []} for key in self.eval_setting}
        self.avg_f1_score = {key: [] for key in self.eval_setting}

    def predict(self, n_frames):
        # random prediction, replace with your model's output
        pred = torch.randint(0, 4, (n_frames,))
        top1_pred_idx = np.random.choice(n_frames, 3, replace=False)  # top1 pred for ini, tran and end state frame idx
        return pred, top1_pred_idx

    def evaluate_one_clip(self, row):
        n_frames = int(row['duration'] * self.fps)
        name = 'novel' if row['is_novel_osc'] else 'known'
        gt = torch.from_numpy(derive_label(row, n_frames=n_frames))
        pred, pred_idx = self.predict(n_frames)

        prec = self.state_prec(pred, gt)
        f1 = self.f1_score(pred, gt)
        unique_labels = torch.unique(gt).cpu().numpy().astype(int)
        unique_labels = unique_labels[unique_labels > 0]
        self.prec_list[name]['avg'].append(prec[unique_labels].mean())
        self.prec_list['all']['avg'].append(prec[unique_labels].mean())
        self.avg_f1_score[name].append(f1[unique_labels].mean())
        self.avg_f1_score['all'].append(f1[unique_labels].mean())
        for label in unique_labels:
            self.prec_list[name][f's{label - 1}'].append(prec[label])
            self.prec_list['all'][f's{label - 1}'].append(prec[label])
        self.state_prec1[name].update(pred_idx, gt)
        self.state_prec1['all'].update(pred_idx, gt)

    def evaluate_one_sc(self, sc):
        df = self.df[self.df['state_transition'] == sc]
        print('*' * 10, f'Evaluating state transition={sc} with {len(df)} clips', '*' * 10)
        for idx, row in df.iterrows():
            self.evaluate_one_clip(row)
        result = np.zeros((6), dtype=float)  # known f1, novel f1, known prec, novel prec, known prec1, novel prec1
        for key in self.eval_setting:
            prec1 = self.state_prec1[key].compute()
            prec_list = self.prec_list[key]
            prec = sum(prec_list['avg']) / len(prec_list['avg'])
            avg_f1_score = self.avg_f1_score[key]
            print(f"Setting {key}: F1 score: {sum(avg_f1_score) / len(avg_f1_score) * 100:.2f} "
                  f"Precision {prec * 100:.2f} | Prec1 {prec1['avg'] * 100:.2f}")
            if key != 'all':
                result_idx = 0 if key == 'known' else 1
                result[result_idx] = sum(avg_f1_score) / len(avg_f1_score)
                result[2 + result_idx] = sum(prec_list['avg']) / len(prec_list['avg'])
                result[4 + result_idx] = prec1['avg']
        print()
        return result

    def evaluate_all(self):
        results = np.zeros((len(self.sc_list), 6), dtype=float)
        for i, sc in enumerate(self.sc_list):
            results[i] = self.evaluate_one_sc(sc)
        print('*' * 20, 'Final Results', '*' * 20)
        avg_results = np.mean(results, axis=0) * 100
        print(f"Known F1 score: {avg_results[0]:.2f} | Novel F1 score: {avg_results[1]:.2f}\n"
              f"Known Precision: {avg_results[2]:.2f} | Novel Precision: {avg_results[3]:.2f}\n"
              f"Known Prec1: {avg_results[4]:.2f} | Novel Prec1: {avg_results[5]:.2f}")


if __name__ == '__main__':
    evaluator = EvalClip('data_files/howtochange_eval.csv')
    evaluator.evaluate_all()