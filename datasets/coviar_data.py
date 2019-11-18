"""
Two types of data are needed.
Type 1: 16 frames from a raw video clip
Type 2: 16 frames from a compressed video(namely 1 P-frame and 15 I-frames)
"""

import os
import os.path
import random

import numpy as np
import cv2
import torch
import torch.utils.data as data
import argparse
from torchvision import transforms
from trans import get_augmentation

from coviar import get_num_frames
from coviar import load


GOP_SIZE = 16

def clip_and_scale(img, size):
    return (img * (127.5 / size)).astype(np.int32)

def get_seg_range(n, num_segments, seg, representation):
    if representation in ['mv', 'residual']:
        n -= 1

    seg_size = float(n-1) / num_segments
    seg_begin = int(np.round(seg_size * seg))
    seg_end = int(np.round(seg_size * (seg+1)))
    if seg_end == seg_begin:
        seg_end = seg_begin + 1

    if representation in ['mv', 'residual']:
        return seg_begin + 1, seg_end + 1

    return seg_begin, seg_end

def get_gop_pos(frame_idx, representation):
    gop_index = frame_idx // GOP_SIZE
    gop_pos = frame_idx % GOP_SIZE
    if representation in ['mv', 'residual']:
        if gop_pos == 0:
            gop_index -= 1
            gop_pos = GOP_SIZE - 1
    else:
        gop_pos = 0
    return gop_index, gop_pos

def color_aug(img, random_h=36, random_l=50, random_s=50):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(float)  # 颜色空间转换函数

    h = (random.random() * 2 - 1.0) * random_h
    l = (random.random() * 2 - 1.0) * random_l
    s = (random.random() * 2 - 1.0) * random_s

    img[..., 0] += h
    img[..., 0] = np.minimum(img[..., 0], 180)

    img[..., 1] += l
    img[..., 1] = np.minimum(img[..., 1], 255)

    img[..., 2] += s
    img[..., 2] = np.minimum(img[..., 2], 255)

    img = np.maximum(img, 0)
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_HLS2BGR)
    return img

class CoviarData(data.Dataset):
    def __init__(self, data_root, data_name,
                 video_list,
                 representation,
                 transform,
                 num_segments,
                 is_train,
                 accumulate):

        self._data_root = data_root
        self._data_name = data_name
        self._num_segments = num_segments
        self._representation = representation
        self._is_train = is_train
        self._accumulate = accumulate
        self._transform = transform
        self.toPIL = transforms.ToPILImage()

        self._input_mean = torch.from_numpy(
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).float()
        self._input_std = torch.from_numpy(
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).float()

        self._load_list(video_list)

    def _load_list(self, video_list):
        self._video_list = []
        with open(video_list, 'r') as f:
            for line in f:
                video, _, label = line.strip().split()
                video_path = os.path.join(self._data_root, video[:-4] + '.mp4')
                self._video_list.append((
                    video_path,
                    int(label),
                    get_num_frames(video_path)))

        print('{} videos loaded.'.format(len(self._video_list)))

    def _get_train_frame_index(self, num_frames, seg):
        # Compute the range of the segment
        seg_begin, seg_end = get_seg_range(num_frames, self._num_segments, seg,
                                                    representation=self._representation)

        # Sample one frame from the segment
        v_frame_idx = random.randint(seg_begin, seg_end-1)
        return get_gop_pos(v_frame_idx, self._representation)

    def _get_test_frame_index(self, num_frames, seg):
        if self._representation in ['mv', 'residual']:
            num_frames -= 1

        seg_size = float(num_frames - 1) / self._num_segments
        v_frame_idx = int(np.round(seg_size * (seg + 0.5)))

        if self._representation in ['mv', 'residual']:
            v_frame_idx += 1

        return get_gop_pos(v_frame_idx, self._representation)


    def __getitem__(self, index):

        if self._representation == 'mv':
            representation_idx = 1
        elif self._representation == 'residual':
            representation_idx = 2
        else:
            representation_idx = 3


        if self._is_train:
            video_path, label, num_frames = random.choice(self._video_list)
        else:
            video_path, label, num_frames = self._video_list[index]

        frames = []
        for seg in range(self._num_segments):

            if self._is_train:
                gop_index, gop_pose = self._get_train_frame_index(num_frames, seg)
            else:
                gop_index, gop_pose = self._get_test_frame_index(num_frames, seg)

            img = load(video_path, gop_index, gop_pose,
                       representation_idx, self._accumulate)

            if img is None:
                print('Error: loading compressed video {} failed.'.format(video_path))
                img = np.zeros((256, 256, 2)) if self._representation == 'mv' else np.zeros((256, 256,3))
            else:
                if self._representation == 'mv':
                    img = clip_and_scale(img, 20)
                    img += 128
                    img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)
                elif self._representation == 'residual':
                    img += 128
                    img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)

            if self._representation == 'iframe':
                img = color_aug(img)
                # BGR to RGB
                img = img[..., ::-1]

            frames.append(img)

        frames = self._transform(frames)

        frames = np.array(frames)
        frames = np.transpose(frames, (0, 3, 1, 2))
        input = torch.from_numpy(frames).float() / 255.0

        if self._representation == 'iframe':
            input = (input - self._input_mean) / self._input_std
        elif self._representation == 'residual':
            input = (input - 0.5) / self._input_std
        elif self._representation == 'mv':
            input = (input - 0.5)

        return input, label


    def __len__(self):
        return len(self._video_list)


if __name__ == '__main__':
    #args = parse_args()
    com = CoviarData('/data2/fb/project/pytorch-coviar-master/data/ucf101/mpeg4_videos',
                     'ucf101',
                     '/data2/fb/project/pytorch-coviar-master/data/datalists/ucf101_split1_train.txt',
                     'residual', get_augmentation(), 4, 1, True)
    train_loader = torch.utils.data.DataLoader(com, batch_size=8,
                                               shuffle=True, num_workers=1,
                                               pin_memory=True)

    for i, input in enumerate(train_loader):
        print('{} : '.format(i))
        print('------------------------------------------------------')
        if i == 5:
            break




