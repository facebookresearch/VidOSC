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
from tqdm import tqdm
from PIL import Image
from data_scripts.read_ann import extract_frames


state_description_mapping = {
    'chopping': ['whole', 'chopping', 'chopped'],
    'slicing': ['whole', 'slicing', 'sliced'],
    'frying': ['raw', 'frying', 'fried'],
    'peeling': ['whole', 'peeling', 'peeled'],
    'blending': ['whole', 'blending', 'blended'],
    'roasting': ['raw', 'roasting', 'roasted'],
    'browning': ['raw', 'browning', 'browned'],
    'grating': ['whole', 'grating', 'grated'],
    'grilling': ['raw', 'grilling', 'grilled'],
    'crushing': ['whole', 'crushing', 'crushed'],
    'melting': ['unmelted', 'melting', 'melted'],
    'squeezing': ['whole', 'squeezing', 'squeezed'],
    'sauteing': ['raw', 'sauteing', 'sauteed'],
    'shredding': ['whole', 'shredding', 'shredded'],
    'whipping': ['liquid', 'whipping', 'whipped'],
    'rolling': ['unrolled', 'rolling', 'rolled'],
    'mashing': ['whole', 'mashing', 'mashed'],
    'mincing': ['whole', 'mincing', 'minced'],
    'coating': ['uncoated', 'coating', 'coated'],
    'zesting': ['whole', 'zesting', 'zested'],
}


class RunClip:
    def __init__(self):
        from transformers import CLIPProcessor, CLIPModel
        model_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
        self.model = CLIPModel.from_pretrained(model_name).cuda()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def inference(self, input_video, text_list, batch_size=64):
        try:
            video_frames = extract_frames(input_video, fps=1)
        except Exception as e:
            print(f'Error extracting frames from {input_video}: {e}')
            return None
        video_frames = video_frames.copy()
        images = [Image.fromarray(frame) for frame in video_frames]
        image_list = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]
        result_list = []
        for image in image_list:
            inputs = self.processor(images=image, text=text_list, return_tensors="pt", padding=True)
            inputs = {name: tensor.cuda() for name, tensor in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits_per_image
            clip_score = (logits / 100).cpu().numpy()
            result_list.append(clip_score)
        clip_score = np.concatenate(result_list, axis=0)
        return clip_score


class RunVideoClip:
    def __init__(self):
        from mmpt.models import MMPTModel
        self.model, self.tokenizer, self.aligner = MMPTModel.from_pretrained("./data_scripts/mmpt_how2.yaml")
        self.model.eval()

    def inference(self, input_video, text_list):
        try:
            video_frames = extract_frames(input_video, fps=30, size=224)
        except Exception as e:
            print(f'Error extracting frames from {input_video}: {e}')
            return None
        video_frames = video_frames.copy()
        num_batches = video_frames.shape[0] // 30
        video_frames = video_frames[:num_batches * 30]
        video_frames = (torch.from_numpy(video_frames.reshape(num_batches, 30, 224, 224, 3)).float() / 255.0)
        n_frames = video_frames.shape[0]
        score = np.zeros((n_frames, len(text_list)))
        for t in tqdm(range(n_frames)):
            frame = video_frames[t].unsqueeze(0).unsqueeze(1)
            for i, text in enumerate(text_list):
                caps, cmasks = self.aligner._build_text_seq(self.tokenizer(text, add_special_tokens=False)["input_ids"])
                caps, cmasks = caps[None, :], cmasks[None, :]
                with torch.no_grad():
                    output = self.model(frame, caps, cmasks, return_score=True)
                score[t, i] = output["score"].item()
        return score


def process(file_name, clip_path, save_path):
    print(f'Processing {file_name} with clip path = {clip_path}, save path = {save_path}')
    df = pd.read_csv(file_name)
    model = RunVideoClip()  # RunClip or RunVideoClip
    for index, row in df.iterrows():
        video_name = os.path.join(clip_path, row['osc'], f"{row['video_id']}_st{row['start_time']}_dur{row['duration']}.mp4")
        pl_name = os.path.join(save_path, row['osc'], f"{row['video_id']}_st{row['start_time']}_dur{row['duration']}.npy")
        os.makedirs(os.path.dirname(pl_name), exist_ok=True)
        if not os.path.exists(video_name):
            print(f"Missing {video_name}")
            continue
        if os.path.exists(pl_name):
            print(f"Already processed {video_name}")
            continue
        verb, object = row['osc'].split('_')
        state_list = state_description_mapping[verb]
        candidate_labels = [state + ' ' + object for state in state_list]
        clip_score = model.inference(video_name, candidate_labels)
        if clip_score is not None:
            np.save(pl_name, clip_score)
            print(f'Saved {pl_name}')


if __name__ == '__main__':
    process('./data_files/howtochange_unlabeled_train.csv', './data/train_clips', './data/train_pseudo_labels/videoclip_probs')