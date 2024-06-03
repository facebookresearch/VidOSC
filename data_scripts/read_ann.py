#
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import os
import ast
import numpy as np
import pandas as pd
import random
import ffmpeg
import time


def extract_frames(video_path, fps, size=None, crop=None, start=None, duration=None):
    if start is not None:
        cmd = ffmpeg.input(video_path, ss=start, t=duration)
    else:
        cmd = ffmpeg.input(video_path)

    if size is None:
        info = [s for s in ffmpeg.probe(video_path)["streams"] if s["codec_type"] == "video"][0]
        size = (info["width"], info["height"])
    elif isinstance(size, int):
        size = (size, size)

    if fps is not None:
        cmd = cmd.filter('fps', fps=fps)
    cmd = cmd.filter('scale', size[0], size[1])

    if crop is not None:
        cmd = cmd.filter('crop', f'in_w-{crop[0]}', f'in_h-{crop[1]}')
        size = (size[0] - crop[0], size[1] - crop[1])

    out = None
    for i in range(5):
        try:
            out, _ = (
                cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True)
            )
            break
        except Exception as e:
            time.sleep(random.random() * 5.)
            if i < 4:
                continue
            print(f"W: FFMPEG file {video_path} read failed!", flush=True)
            if isinstance(e, ffmpeg.Error):
                print("STDOUT:", e.stdout, flush=True)
                print("STDERR:", e.stderr, flush=True)
            # raise

    if out is None:
        return None
    video = np.frombuffer(out, np.uint8).reshape([-1, size[1], size[0], 3])
    return video


def derive_label(annotation, n_frames):
    state_to_label = {
        'initial_state': 1,
        'transitioning_state': 2,
        'end_state': 3,
    }
    gt = np.zeros(n_frames)
    for state in ['initial_state', 'transitioning_state', 'end_state']:
        for time_range in ast.literal_eval(annotation[state]):
            start, end = time_range
            # print(f"state={state}, start={start}s, end={end}s")
            gt[round(start):round(end)] = state_to_label[state]
    return gt


def read_data(ann_file, clip_dir):
    df = pd.read_csv(ann_file)
    print(f'loaded {len(df)} rows from {ann_file}')  # HowToChange (eval) set
    for i, row in df.iterrows():
        clip_file = os.path.join(clip_dir, row['osc'], f"{row['video_id']}_st{row['start_time']}_dur{row['duration']}.mp4")
        assert os.path.exists(clip_file), f'{clip_file} does not exist'
        print('-' * 20, f"{clip_file} ({row['osc']})", '-' * 20)

        # extract frames ar 1fps
        frames = extract_frames(clip_file, fps=1)
        print(f'extracted {len(frames)} frames, shape {frames.shape}')

        # derive annotation at 1fps
        gt_label = derive_label(row, n_frames=len(frames))
        print('ground truth state label (0=bg, 1=ini, 2=tran, 3=end):', gt_label)


if __name__ == '__main__':
    read_data('./data_files/howtochange_eval.csv', './data/eval_clips')