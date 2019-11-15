"""
Two types of data are needed.
Type 1: 16 frames from a raw video clip
Type 2: 16 frames from a compressed video(namely 1 P-frame and 15 I-frames)
"""

import os
import torch.utils.data as data
import cv2
import sys
sys.path.append('..')
import random
import skvideo.io
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import numpy as np
import pandas as pd
import argparse

ens = os.environ

class RawData(data.Dataset):

    def __init__(self, root, mode='train', args=None):

        self.transforms = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop(112),
            transforms.ToTensor()
        ])

        self.root = root
        self.mode = mode
        self.args = args
        self.toPIL = transforms.ToPILImage()
        self.tensortrans = transforms.Compose([transforms.ToTensor()])

        self.split = '1'

        train_split_path = os.path.join(root, 'split', 'trainlist0' + self.split + '.txt')
        self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        test_split_path = os.path.join(root, 'split', 'testlist0' + self.split + '.txt')
        self.test_split = pd.read_csv(test_split_path, header=None, sep=' ')[0]
        if mode == 'train':
            self.list = self.train_split
        else:
            self.list = self.test_split


    def __getitem__(self, index):
        #index = index.item()

        if self.args.msr and self.mode == 'train':
            videodata = self.loadcvvideo_msr(index)
        else:
            videodata = self.loadcvvideo(index)

        clip1 = self.crop(videodata[0:16])
        clip2 = self.crop(videodata[16:32])
        clip3 = self.crop(videodata[32:48])

        if not self.args.vcop:
            seed = random.randint(0, 2)
            if seed == 0:
                return clip1, clip3, clip2
            if seed == 1:
                return clip1, clip2, clip3
            if seed == 2:
                return clip3, clip2, clip1
        else:
            if self.args.num_order == 4:
                seed = random.randint(0, 3)
                if seed == 0:
                    return clip1, clip2, clip3, torch.tensor(seed)
                if seed == 1:
                    return clip2, clip1, clip3, torch.tensor(seed)
                if seed == 2:
                    return clip1, clip3, clip2, torch.tensor(seed)
                if seed == 3:
                    return clip3, clip1, clip2, torch.tensor(seed)
            if self.args.num_order == 2:
                seed = random.randint(0, 1)
                if seed == 0:
                    return clip1, clip2, torch.tensor(seed)
                if seed == 1:
                    return clip3, clip1, torch.tensor(seed)


    def loadcvvideo(self, index):
        need = 48
        fname = self.list[index]
        fname = os.path.join(self.root, 'video', fname)

        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        while frame_count < need:
            index = np.random.randint(self.__len__())
            fname = self.list[index]
            fname = os.path.join(self.root, 'video', fname)

            capture = cv2.VideoCapture(fname)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        start =np.random.randint(0, frame_count - need + 1)
        buffer = []
        count = 0
        retaining = True
        sample_count = 0

        while(sample_count < need and retaining):
            retaining, frame = capture.read()

            if retaining is False:
                count += 1
                break

            if count >= start:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                buffer.append(frame)
                sample_count = sample_count + 1

            count += 1

        capture.release()
        while len(buffer) < need:
            index = np.random.randint(self.__len__())

            buffer = self.loadcvvideo(index)
            print('reload')

        return buffer


    def loadcvvideo_msr(self, index):
        fname = self.list[index]
        fname = os.path.join(self.root, 'video', fname)

        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate = np.random.randint(low=1, high=4)

        shortest_len = (48-1) * sample_rate + 1 + 1
        while frame_count < shortest_len:
            index = np.random.randint(self.__len__())
            fname = self.list[index]
            fname = os.path.join(self.root, 'video', fname)

            capture = cv2.VideoCapture(fname)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        start = np.random.randint(0, frame_count - shortest_len + 1)

        if start > 0:
            start = start - 1
        buffer = []
        count = 0
        retaining = True
        sample_count = 0

        while (sample_count < 48 and retaining):
            retaining, frame = capture.read()

            if retaining is False:
                count += 1
                break

            if count >= start and (count - start) % sample_rate == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                buffer.append(frame)
                sample_count = sample_count + 1
            count += 1

        capture.release()
        while len(buffer) < 48:
            index = np.random.randint(self.__len__())

            buffer = self.loadcvvideo_msr(index)
            print('reload')

        return buffer, sample_rate


    def crop(self, frames):
        video_clip = []
        seed = random.random()

        for frame in frames:
            random.seed(seed)
            frame = self.toPIL(frame)
            frame = self.transforms(frame)

            video_clip.append(frame)

        clip = torch.stack(video_clip).permute(1, 0, 2, 3)

        return clip


    def __len__(self):
        return len(self.list)


def parse_args():

    parser = argparse.ArgumentParser(description='Video Clip Reconstruction and Order Prediction')
    parser.add_argument('--lpls', type=bool, default=False, help='use lpls loss or not')
    parser.add_argument('--msr', type=bool, default=False, help='use multi sample rate or not')
    parser.add_argument('--vcop', type=bool, default=True, help='predict video clip or not')
    parser.add_argument('--num_order', type=int, default=2, help='number of video clip order to predict')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')
    parser.add_argument('--epochs', type=int, default=300, help='number of total epochs to run')
    parser.add_argument('--exp_name', type=str, default='default', help='experiment name')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    com = RawData("/data2/video_data/UCF-101", mode='train', args=args)
    train_dataloader = DataLoader(com, batch_size=8, num_workers=1, shuffle=False)
    # DataLoader return should be like : Input, Label
    # clip1, clip2??
    for i, (clip1, clip2, a) in enumerate(train_dataloader):
        print('{} :'.format(i))
        print('clip1.size: {}'.format(clip1.size()))
        print('clip2.size: {}'.format(clip2.size()))
        print('a: {}'.format(a))
        print('---------------------------------------------------------------------')
        if i == 5:
            break


