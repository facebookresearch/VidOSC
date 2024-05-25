#
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import pandas as pd
import subprocess
import os


def download_and_crop_video(video_id, start_time, duration, save_path):
    try:
        video_name = os.path.join(save_path, f"{video_id}.mp4")
        clip_name = os.path.join(save_path, f"{video_id}_st{start_time}_dur{duration}.mp4")
        command_download = f"yt-dlp 'https://www.youtube.com/watch?v={video_id}' -o {video_name} -f 'bestvideo[ext=mp4][height<=480]'"
        subprocess.call(command_download, shell=True)
        command_clip = f"ffmpeg -y -ss {start_time} -i '{video_name}' -to {duration} -c:v libx264 '{clip_name}'"
        subprocess.call(command_clip, shell=True)
        print(f"Processed {video_id} successfully.")
        os.remove(video_name)
    except Exception as e:
        print(f"Failed to process {video_id}: {e}")


def preprocess_video(csv_file, save_dir):
    print(f'Processing {csv_file}, saving to {save_dir}')
    df = pd.read_csv(csv_file)
    for index, row in df.iterrows():
        save_path = os.path.join(save_dir, row['osc'])
        os.makedirs(save_path, exist_ok=True)
        download_and_crop_video(row['video_id'], row['start_time'], row['duration'], save_path)


if __name__ == '__main__':
    preprocess_video('./data_files/howtochange_eval.csv', './data/eval_clips')
    preprocess_video('./data_files/howtochange_unlabeled_train.csv', './data/train_clips')