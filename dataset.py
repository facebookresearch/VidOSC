#
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
import os
import torch
import numpy as np
import pandas as pd
import ast
from torch.utils.data import Dataset
from data_scripts.read_ann import derive_label


def build_vocab(args):
    df = pd.read_csv(os.path.join(args.ann_dir, 'howtochange_eval.csv'))
    df['verb'] = df['osc'].apply(lambda x: x.split('_')[0])
    if 'all' not in args.sc_list:
        df = df[df['verb'].isin(args.sc_list)]
    vocab = {'background': 0}
    key = 'verb'
    for i, k in enumerate(df[key].unique()):
        vocab[k] = i + 1
    sc_list = df['verb'].unique().tolist()
    print(f"Vocab len: {len(vocab)}")
    print(f"Vocab {vocab}")
    print(f"State Transition {sc_list}")
    return vocab, sc_list, df


class HowToChangeFeatDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.feat_dir = args.feat_dir
        self.vocab, _, self.df = build_vocab(args)
        print(f"HowToChange Eval: state transition = {args.sc_list} -> {len(self.df)} videos")
        self.max_seq_len = int(self.df['duration'].max())
        print(f"Max sequence length: {self.max_seq_len}")

    def load_feat(self, row):
        video_id = row['video_id']
        feat_path = os.path.join(self.feat_dir, 'feats', video_id + '.pth.tar')
        feat = torch.load(feat_path)  # note: feat_path are NOT truncated based on start time and duration, so we need to truncate it here, modify if needed

        start = float(row['start_time'])
        duration = float(row['duration'])
        end = min(start + duration, feat.shape[0])
        feat = feat[int(start):int(end)]

        if self.args.det > 0:  # + object-centric feature
            obj_features = torch.zeros_like(feat)
            file_name = row['video_name'] + '_obj.pth.tar'
            obj_feat_path = os.path.join(self.feat_dir, 'feats_handobj', row['osc'], file_name)
            
            if os.path.exists(obj_feat_path):
                # obj_features = torch.load(obj_feat_path)
                obj_feat = torch.load(obj_feat_path)
                obj_idx = np.load(obj_feat_path.replace('.pth.tar', '.npy'))
                obj_idx = obj_idx[obj_idx < len(obj_features)]
                obj_features[obj_idx] = obj_feat[0:len(obj_idx)]
            else:
                print(f'Warning! {obj_feat_path} do not exist')
            feat = torch.cat((feat, obj_features), dim=-1)

        return feat

    def derive_label(self, annotation, n_frames):
        gt = np.zeros(n_frames)
        for state in ['s0', 's1', 's2']:
            for time_range in ast.literal_eval(annotation[state]):
                start, end = time_range
                gt[round(start):round(end)] = int(state[-1]) + 1
        return gt

    def __getitem__(self, index):
        row = self.df.iloc[index]
        feat = self.load_feat(row)
        label = torch.from_numpy(derive_label(row, feat.shape[0]))
        return feat, label, row['osc'], row['is_novel_osc']

    def __len__(self):
        return len(self.df)



class HowToChangeFeatCLIPLabelDataset(HowToChangeFeatDataset):
    def __init__(self, args):
        super().__init__(args)
        self.load_data()
    
    def load_data(self):
        df = pd.read_csv(os.path.join(self.args.ann_dir, 'howtochange_unlabeled_train.csv'))
        df['verb'] = df['osc'].apply(lambda x: x.split('_')[0])
        if 'all' not in self.args.sc_list:
            df = df[df['verb'].isin(self.args.sc_list)]
        print(f"HowToChange Train: state transition = {self.args.sc_list} -> {len(df)} videos")
        self.max_seq_len = int(df['duration'].max())
        self.data_list = []
        for i, row in df.iterrows():
            pl_path = os.path.join(self.args.pseudolabel_dir, row['osc'], row['video_name'] + '.npz')
            if not os.path.exists(pl_path):
                print(f'Missing pseudo label {pl_path}')
                continue
            pl = np.load(pl_path)['arr_0']
            self.data_list.append({
                'row': row,
                'pseudo_label': pl
            })
        print(f"{len(self.data_list)} training clips loaded")
    
    def pad_sequence(self, feat):
        t, dim = feat.shape
        padded_feat = torch.cat((feat, torch.zeros((self.max_seq_len - t, dim))), dim=0)
        return padded_feat

    def pad_label(self, label):
        t = label.shape[0]
        padded_label = torch.cat((label, -torch.ones(self.max_seq_len - t)), dim=0).long()
        return padded_label
    
    def __getitem__(self, index):
        data_dict = self.data_list[index]
        feat = self.load_feat(data_dict['row'])
        pseudo_label = torch.from_numpy(data_dict['pseudo_label'])
        pseudo_label = pseudo_label[0: feat.shape[0]]
        feat = self.pad_sequence(feat)
        pseudo_label = self.pad_label(pseudo_label)
        return feat, pseudo_label

    def __len__(self):
        return len(self.data_list)


# Deprecated: generate pseudo labels based on clip / video clip similarity score
# VideoCLIP threshold
sc_to_threshold_a = {
    'chopping': [6, 0.05],
    'slicing': [12, 0],
    'frying': [10, 0.05],
    'peeling': [8, 0],
    'blending': [8, 0.1],
    'roasting': [12, 0.05],
    'browning': [6, 0.05],
    'grating': [12, 0.2],
    'grilling': [6, 0],
    'crushing': [8, 0.2],
    'melting': [6, 0],
    'squeezing': [8, 0],
    'sauteing': [6, 0.05],
    'shredding': [6, 0],
    'whipping': [12, 0],
    'rolling': [10, 0],
    'mashing': [6, 0.05],
    'mincing': [8, 0],
    'coating': [8, 0.05],
    'zesting': [12, 0]
}


# CLIP threshold
sc_to_threshold_b = {
    'chopping': [0.35, 0.38],
    'slicing': [0.39, 0.35],
    'frying': [0.35, 0.43],
    'peeling': [0.35, 0.35],
    'blending': [0.36, 0.43],
    'roasting': [0.35, 0.37],
    'browning': [0.37, 0.37],
    'grating': [0.35, 0.43],
    'grilling': [0.35, 0.37],
    'crushing': [0.35, 0.38],
    'melting': [0.35, 0.36],
    'squeezing': [0.35, 0.35],
    'sauteing': [0.35, 0.45],
    'shredding': [0.36, 0.40],
    'whipping': [0.39, 0.38],
    'rolling': [0.36, 0.36],
    'mashing': [0.36, 0.37],
    'mincing': [0.35, 0.35],
    'coating': [0.38, 0.36],
    'zesting': [0.35, 0.36]
}


class HowToChangeFeatCLIPLabelDatasetDeprecated(HowToChangeFeatDataset):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        if args.use_videoclip:
            self.mode = 'a'
            self.pl_dir = os.path.join(args.pseudolabel_dir, 'videoclip_probs')
        else:
            self.mode = 'b'
            self.bg_thre = 0.5
            self.pl_dir = os.path.join(args.pseudolabel_dir, 'clip_probs')
        self.ordering = True  # enforce causal ordering
        self.pseudo_label_stat()

    def pseudo_label_stat(self):
        df = pd.read_csv(os.path.join(self.args.ann_dir, 'howtochange_unlabeled_train.csv'))
        df['verb'] = df['osc'].apply(lambda x: x.split('_')[0])
        if 'all' not in self.args.sc_list:
            df = df[df['verb'].isin(self.args.sc_list)]
        print(f"HowToChange Train: state transition = {self.args.sc_list} -> {len(df)} videos")
        self.max_seq_len = int(df['duration'].max())

        self.data_list = []
        self.state_cnt = [0, 0, 0, 0]
        skip_cnt = 0
        for i, row in df.iterrows():
            file_name = row['video_id'] + '_st' + str(row['start_time']) + '_dur' + str(row['duration']) + '.npy' # modify accordingly
            pl_path = os.path.join(self.pl_dir, row['osc'], file_name)  # pseudo label path
            if not os.path.exists(pl_path):
                skip_cnt += 1
                # print(f'Missing {clip_path}')
                continue
            clip_score = np.load(pl_path)

            if self.mode == 'a':
                threshold = sc_to_threshold_a[row['osc'].split('_')[0]]
                self.bg_thre = threshold[0]
            else:
                threshold = sc_to_threshold_b[row['osc'].split('_')[0]]

            if np.all(clip_score.sum(axis=1) < self.bg_thre):
                # print(f"Skip {row['video_path']} due to low clip score")
                continue

            self.data_list.append({
                'row': row,
                'pseudo_label': self.derive_pseudo_label(clip_score, row['osc'], threshold),
            })
        print(f'{len(self.df)} videos loaded, {skip_cnt} does not have clip np file, '
            f'after filtering: {len(self.data_list)}')
        print(f"Pseudo-labeled State count: {self.state_cnt}, ratio {np.array(self.state_cnt) / self.state_cnt[-1]}")

    def enforce_two_set_ordering(self, s0, s1):
        while len(s0) > 0 and len(s1) > 0 and s0[-1] >= s1[0]:
            s0_diff = s0[-1] - np.mean(s0)
            s1_diff = np.mean(s1) - s1[0]
            if s0_diff > s1_diff:  # remove s0
                s0 = s0[:-1]
            else:
                s1 = s1[1:]
        return s0, s1

    def causal_ordering(self, s0_cond, s1_cond, s2_cond):
        s0_cond_prev = np.where(s0_cond)[0]
        s1_cond_prev = np.where(s1_cond)[0]
        s2_cond_prev = np.where(s2_cond)[0]
        # print('before', s0_cond_prev, s1_cond_prev, s2_cond_prev)
        s0_cond, s1_cond = self.enforce_two_set_ordering(s0_cond_prev, s1_cond_prev)
        s1_cond, s2_cond = self.enforce_two_set_ordering(s1_cond, s2_cond_prev)
        # print('after', s0_cond, s1_cond, s2_cond)
        return s0_cond, s1_cond, s2_cond

    def derive_pseudo_label(self, score, osc, threshold):
        if self.mode == 'a':
            bg_thre, margin = threshold
            bg_cond = np.sum(score, axis=1) < bg_thre
            s0_cond = ~bg_cond & (score[:, 0] - score[:, 1] > margin) & (score[:, 0] - score[:, 2] > margin)
            s1_cond = ~bg_cond & (score[:, 1] - score[:, 0] > margin) & (score[:, 1] - score[:, 2] > margin)
            s2_cond = ~bg_cond & (score[:, 2] - score[:, 0] > margin) & (score[:, 2] - score[:, 1] > margin)
        else:
            thre1, thre2 = threshold
            bg_cond = np.sum(score, axis=1) < self.bg_thre
            s0_cond = ~bg_cond & (score[:, 0] > score[:, 2]) & (score[:, 0] > thre1)
            s1_cond = ~bg_cond & (score[:, 1] > score[:, 0]) & (score[:, 1] > score[:, 2])
            s2_cond = ~bg_cond & (score[:, 2] > score[:, 0]) & (score[:, 2] > thre2)

        if self.ordering:
            s0_cond, s1_cond, s2_cond = self.causal_ordering(s0_cond, s1_cond, s2_cond)
        else:
            s0_cond = np.where(s0_cond)[0]
            s1_cond = np.where(s1_cond)[0]
            s2_cond = np.where(s2_cond)[0]

        pseudo_label = np.ones(score.shape[0]) * -1
        key = osc.split('_')[0]
        category_id = self.vocab[key]
        assert category_id > 0
        pseudo_label[bg_cond] = 0
        pseudo_label[s0_cond] = 3 * category_id - 2
        pseudo_label[s1_cond] = 3 * category_id - 1
        pseudo_label[s2_cond] = 3 * category_id

        self.state_cnt[0] += len(s0_cond)
        self.state_cnt[1] += len(s1_cond)
        self.state_cnt[2] += len(s2_cond)
        self.state_cnt[3] += len(score)
        return pseudo_label

    def pad_sequence(self, feat):
        t, dim = feat.shape
        padded_feat = torch.cat((feat, torch.zeros((self.max_seq_len - t, dim))), dim=0)
        return padded_feat

    def pad_label(self, label):
        t = label.shape[0]
        padded_label = torch.cat((label, -torch.ones(self.max_seq_len - t)), dim=0).long()
        return padded_label

    def __getitem__(self, index):
        data_dict = self.data_list[index]
        feat = self.load_feat(data_dict['row'])
        pseudo_label = torch.from_numpy(data_dict['pseudo_label'])
        pseudo_label = pseudo_label[0: feat.shape[0]]
        feat = self.pad_sequence(feat)
        pseudo_label = self.pad_label(pseudo_label)
        return feat, pseudo_label

    def __len__(self):
        return len(self.data_list)

