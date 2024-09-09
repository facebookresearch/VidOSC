#
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import os
import glob

import pandas as pd
import torch
import pickle
import random
import itertools
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
from torchtext.vocab import vocab


class ChangeItFeatDataset(Dataset):
    def __init__(self, args, split):
        self.args = args
        self.padding = True
        self.use_vocab = True  # use shared vocab
        self.max_seq_len = 900
        
        self.feat_dim = 768 * (1 + args.det)     # internvideo feat
        pickle_roots = [os.path.join(args.dataset_root, 'feats_internvideo')] 
        self.feat_obj_root = os.path.join(args.dataset_root, 'feats_obj')

        annotation_root = os.path.join(args.dataset_root, "annotations")
        noise_adapt_weight_root = None if args.ignore_video_weight else os.path.join(args.dataset_root, "videos")
        noise_adapt_weight_threshold_file = None if args.ignore_video_weight else os.path.join(args.dataset_root, "categories.csv")

        if args.novel_obj:  # changeit open-world
            split_df = pd.read_csv(os.path.join(args.dataset_root, 'open_world_split.csv'))
            seen_obj_classes = split_df['seen_classes'].tolist()
            unseen_obj_classes = split_df['unseen_classes'].tolist()
            seen_obj_classes = [x for x in seen_obj_classes if not pd.isna(x)]
            unseen_obj_classes = [x for x in unseen_obj_classes if not pd.isna(x)]
            if split in ['train', 'val']:
                self.classes = {x: i for i, x in enumerate(seen_obj_classes)}
            else:  # novel obj - test split
                self.classes = {x: i for i, x in enumerate(unseen_obj_classes)}
        elif args.category is not None:  # single-task
            self.classes = {args.category: 0}
        else: # multi-task
            self.classes = {x: i for i, x in enumerate(sorted(set([os.path.basename(fn) for fn in itertools.chain(*[
                glob.glob(os.path.join(root, "*")) for root in pickle_roots
            ]) if os.path.isdir(fn)])))}

        self.n_classes = len(self.classes)
        self.classid2name = {v: k for k, v in self.classes.items()}
        print(f'{self.n_classes} dataset classes {self.classes}')

        self.build_vocab(args)
        if split == 'test':
            split = 'val'

        self.files = {key: sorted(itertools.chain(*[
            glob.glob(os.path.join(root, key, "*.pth.tar")) for root in pickle_roots
        ])) for key in self.classes.keys()}

        self.annotations = {key: {
            os.path.basename(fn).split(".")[0]: np.uint8(
                [int(line.strip().split(",")[1]) for line in open(fn).readlines()])
            for fn in glob.glob(os.path.join(annotation_root, key, "*.csv"))
        } for key in self.classes.keys()} if annotation_root is not None else None

        if split == "train":
            for key in self.classes.keys():
                for fn in self.files[key].copy():
                    if os.path.basename(fn).split(".")[0] in self.annotations[key]:
                        self.files[key].remove(fn)
        elif split == "val":
            for key in self.classes.keys():
                for fn in self.files[key].copy():
                    if os.path.basename(fn).split(".")[0] not in self.annotations[key]:
                        self.files[key].remove(fn)

        self.flattened_files = []
        for key in self.classes.keys():
            self.flattened_files.extend([(key, fn) for fn in self.files[key]])

        # Noise adaptive weighting
        if noise_adapt_weight_root is None:
            return
        self.noise_adapt_weight = {}
        for key in self.classes.keys():
            with open(os.path.join(noise_adapt_weight_root, f"{key}.csv"), "r") as f:
                for line in f.readlines():
                    vid_id, score = line.strip().split(",")
                    self.noise_adapt_weight[vid_id] = float(score)
        self.noise_adapt_weight_thr = {line.split(",")[0]: float(line.split(",")[2].strip())
                                for line in open(noise_adapt_weight_threshold_file, "r").readlines()[1:]}

    def build_vocab(self, args):
        self.state_id_mapping = {}
        self.action_id_mapping = {}
        self.object_mapping = {}
        self.action_to_state = {}

        state_vocab_df = pd.read_csv(os.path.join(args.dataset_root, 'state_description_vocab.csv'))
        filtered_df = state_vocab_df[state_vocab_df['Dir_Name'].isin(self.classes)]
        self.object_vocab = vocab(Counter(filtered_df['Object'].tolist()))
        self.object_classes = len(self.object_vocab)
        print(f'object vocab: {self.object_vocab.get_itos()}')
        for i, row in filtered_df.iterrows():
            obj_name = row['Dir_Name']
            self.object_mapping[obj_name] = self.object_vocab.get_stoi()[row['Object']]

        if self.use_vocab:  # shared vocab
            s0_key = 'Initial_State'
            s1_key = 'End_State'
            state_list = ['background'] + filtered_df[s0_key].tolist() + filtered_df[s1_key].tolist()
            action_list = ['background'] + filtered_df['Action'].tolist()
            self.state_vocab = vocab(Counter(state_list))
            self.action_vocab = vocab(Counter(action_list))
            print(f'state vocab: {self.state_vocab.get_itos()}\n'
                f'action vocab: {self.action_vocab.get_itos()}')

            self.state_classes = len(self.state_vocab)
            self.action_classes = len(self.action_vocab)

            self.state_bg_idx = self.state_vocab.get_stoi()['background']
            self.action_bg_idx = self.action_vocab.get_stoi()['background']
            for i, row in filtered_df.iterrows():
                obj_name = row['Dir_Name']
                class_id = self.classes[obj_name]
                ini_state = row['Initial_State']
                end_state = row['End_State']
                s0_id = self.state_vocab.get_stoi()[ini_state]
                s1_id = self.state_vocab.get_stoi()[end_state]
                action_id = self.action_vocab.get_stoi()[row['Action']]

                self.state_id_mapping[2 * class_id] = s0_id
                self.state_id_mapping[2 * class_id + 1] = s1_id
                self.action_id_mapping[class_id] = action_id
                self.action_to_state[action_id] = (s0_id, s1_id)

        else:
            self.state_classes = self.n_classes * 2 + 1
            self.action_classes = self.n_classes + 1
            self.state_bg_idx = self.n_classes * 2
            self.action_bg_idx = self.n_classes
            for obj, class_id in self.classes.items():
                # identity mapping
                self.state_id_mapping[2 * class_id] = 2 * class_id
                self.state_id_mapping[2 * class_id + 1] = 2 * class_id + 1
                self.action_id_mapping[class_id] = class_id
                self.action_to_state[class_id] = (2 * class_id, 2 * class_id + 1)

        print(f'{self.state_classes} state classes | {self.action_classes} action classes\n' 
            f'State bg idx {self.state_bg_idx} | Action bg idx {self.action_bg_idx}\n'
            f'State id mapping {self.state_id_mapping}\nAction id mapping {self.action_id_mapping}\n'
            f'Action to state {self.action_to_state}')

    def get_state_action_mapping(self):
        vocab_size = len(self.state_vocab) + len(self.action_vocab)
        return self.feat_dim, self.n_classes, self.classid2name, self.state_classes, self.action_classes, self.object_classes, \
            self.state_id_mapping, self.action_id_mapping, self.action_to_state, vocab_size

    def load_feat(self, pickle_fn):
        video_features = torch.load(pickle_fn)
        obj_features = torch.zeros_like(video_features)
        if self.args.det == 0:
            return video_features
        else:
            class_name = pickle_fn.split('/')[-2]
            file_id = os.path.basename(pickle_fn).split(".")[0]
            obj_feat_file = os.path.join(self.feat_obj_root, class_name, file_id + "_obj.pth.tar")
            if os.path.exists(obj_feat_file):
                obj_feat = torch.load(obj_feat_file)
                obj_idx = np.load(os.path.join(self.feat_obj_root, class_name, file_id + "_obj.npy"))
                obj_features[obj_idx] = obj_feat
            video_features = torch.cat((video_features, obj_features), dim=-1)
            return video_features
    
    def pad_sequence(self, feat):
        t, *dims = feat.shape
        padded_feat = torch.cat((feat, torch.zeros([self.max_seq_len - t, *dims], dtype=feat.dtype)), dim=0)
        return padded_feat
    
    def __getitem__(self, idx):
        class_name, pickle_fn = self.flattened_files[idx]
        file_id = os.path.basename(pickle_fn).split(".")[0]
        video_features = self.load_feat(pickle_fn)
        if self.padding:
            video_features = self.pad_sequence(video_features)
        annotation = self.annotations[class_name][file_id] \
            if self.annotations is not None and file_id in self.annotations[class_name] else None
        video_level_score = self.noise_adapt_weight[file_id] - self.noise_adapt_weight_thr[class_name] \
            if hasattr(self, "noise_adapt_weight") else 1.0
            
        return class_name + "/" + file_id, self.classes[
            class_name], video_features, annotation, video_level_score  #
    
    @property
    def n_classes(self):
        return len(self.classes)

    def __len__(self):
        return len(self.flattened_files)

    def __repr__(self):
        string = f"ChangeItDataset(n_classes: {self.n_classes}, n_samples: {self.__len__()}, " \
                f"deterministic: {self.deterministic})"
        for key in sorted(self.classes.keys()):
            string += f"\n> {key:20} {len(self.files[key]):4d}"
            if hasattr(self, "noise_adapt_weight_thr"):
                string += f" (above threshold {self.noise_adapt_weight_thr[key]:.3f}: " \
                        f"{len([fn for fn in self.files[key] if self.noise_adapt_weight[os.path.basename(fn).split('.')[0]] > self.noise_adapt_weight_thr[key]]):4d})"
        return string

    @n_classes.setter
    def n_classes(self, value):
        self._n_classes = value
        
        
class ChangeItFeatCLIPLabelDataset(ChangeItFeatDataset):
    def __init__(self, args, split):
        super().__init__(args, split)
        self.clip_probs_root = os.path.join(self.args.dataset_root, "clip_probs")

        self.bg_thre = args.bg_thre
        self.state_thre1 = args.state_thre1
        self.state_thre2 = args.state_thre2
        self.action_thre = args.action_thre

        self.action_as_state_neg = True
        self.state_as_action_neg = True     
        self.action_ordering = True        
        
        self.pseudo_label_stat()

    def __getitem__(self, idx):
        class_name, pickle_fn = self.flattened_files[idx]
        file_id = os.path.basename(pickle_fn).split(".")[0]

        video_features = self.load_feat(pickle_fn)
        state_label, action_label = self.psuedo_label_dict[class_name + "/" + file_id]
        feat, state_label, action_label = self.pad_sequence(video_features, state_label, action_label)

        video_level_score = 1.0
        if hasattr(self, "noise_adapt_weight"):
            video_level_score = self.noise_adapt_weight[file_id] - self.noise_adapt_weight_thr[class_name]
            video_level_score = 1 / (1 + np.exp(-1000 * video_level_score))
        return feat, state_label, action_label, video_level_score

    def pad_sequence(self, feat, label1, label2):
        t, *dims = feat.shape
        if t != label1.shape[0]:
            label1 = label1[:t]
            label2 = label2[:t]
        padded_feat = torch.cat((feat, torch.zeros([self.max_seq_len - t, *dims], dtype=feat.dtype)), dim=0)
        padded_label1 = torch.cat((torch.from_numpy(label1), -torch.ones(self.max_seq_len - t)), dim=0).long()
        padded_label2 = torch.cat((torch.from_numpy(label2), -torch.ones(self.max_seq_len - t)), dim=0).long()
        return padded_feat, padded_label1, padded_label2

    def casual_ordering(self, s0, s1):
        while len(s0) > 0 and len(s1) > 0 and s0[-1] >= s1[0]:
            s0_diff = s0[-1] - np.mean(s0)
            s1_diff = np.mean(s1) - s1[0]
            if s0_diff > s1_diff:  # remove s0
                s0 = s0[:-1]
            else:
                s1 = s1[1:]
        return s0, s1

    def get_label(self, class_id, prob):
        state_label = -np.ones(prob.shape[0], dtype=int)
        action_label = -np.ones(prob.shape[0], dtype=int)

        bg_cond = np.sum(prob, axis=1) < self.bg_thre
        s0_cond = (prob[:, 0] > self.state_thre1) & (prob[:, 0] > prob[:, 1])
        s1_cond = (prob[:, 1] > self.state_thre2) & (prob[:, 1] > prob[:, 0])
        action_cond = (prob[:, 2] > self.action_thre) & (prob[:, 2] > prob[:, 0]) & (prob[:, 2] > prob[:, 1])

        s0_cond_prev = np.where(s0_cond)[0]
        s1_cond_prev = np.where(s1_cond)[0]
        action_cond_prev = np.where(action_cond)[0]
        s0_cond, action_cond_prev2 = self.casual_ordering(s0_cond_prev, action_cond_prev)
        action_cond, s1_cond = self.casual_ordering(action_cond_prev2, s1_cond_prev)
        s0_cond, s1_cond = self.casual_ordering(s0_cond, s1_cond)

        self.bg_count += bg_cond.sum()
        self.s0_count += len(s0_cond)
        self.s1_count += len(s1_cond)
        self.action_cnt += len(action_cond)
        self.all_count += len(prob)

        state_label[bg_cond] = self.state_bg_idx
        if self.action_as_state_neg:
            state_label[action_cond] = self.state_bg_idx  # action as state negatives
        state_label[s0_cond] = self.state_id_mapping[2 * class_id]
        state_label[s1_cond] = self.state_id_mapping[2 * class_id + 1]

        action_label[bg_cond] = self.action_bg_idx
        if self.state_as_action_neg:
            action_label[s0_cond] = self.action_bg_idx
            action_label[s1_cond] = self.action_bg_idx
        if self.action_ordering:
            action_label[action_cond] = self.action_id_mapping[class_id]
        else:
            action_label[action_cond_prev] = self.action_id_mapping[class_id]

        return state_label, action_label

    def pseudo_label_stat(self):
        self.psuedo_label_dict = {}
        self.bg_count, self.s0_count, self.s1_count, self.action_cnt, self.all_count = 0, 0, 0, 0, 0

        for idx, (class_name, video_fn) in enumerate(self.flattened_files):
            file_id = os.path.basename(video_fn).split(".")[0]
            np_file = os.path.join(self.clip_probs_root, class_name, file_id + ".npy")
            assert os.path.exists(np_file)
            clip_probs = np.load(np_file)
            label1, label2 = self.get_label(self.classes[class_name], clip_probs)
            self.psuedo_label_dict[class_name + "/" + file_id] = [label1, label2]

        print(f'{self.all_count} frames, {self.bg_count} ({self.bg_count / self.all_count * 100:.2f}%) is background, '
              f'{self.s0_count} ({self.s0_count / self.all_count * 100:.2f}%) is state 0, '
              f'{self.s1_count} ({self.s1_count / self.all_count * 100:.2f}%) is state 1, '
              f'{self.action_cnt} ({self.action_cnt / self.all_count * 100:.2f}%) is action')
        print('-' * 80)