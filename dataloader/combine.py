import torch.utils.data as data
from torch.utils.data import DataLoader
import sys
sys.path.append('../')
from trans import get_augmentation
from datasets.coviar_data import CoviarData
from datasets.raw_data import ClassifyDataSet
from config import params
import random

class combine_data(data.Dataset):
    def __init__(self):
        self.data = ClassifyDataSet("/data2/video_data/UCF-101", mode='train')

    def __getitem__(self, index):
        coviar_data = CoviarData(params['mpeg_data'],
                                 'ucf101',
                                 params['mpeg_video_list'],
                                 'residual', get_augmentation(), 16, True, True).__getitem__(index)

        raw_data = ClassifyDataSet("/data2/video_data/UCF-101", mode='train').__getitem__(index)

        if coviar_data[1] == raw_data[1]:
            label = coviar_data[1]
        else:
            print("label not same!")
            label = -1

        return coviar_data[0], raw_data[0], label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    temp = combine_data()
    train_loader = DataLoader(temp, batch_size=1, shuffle=True, num_workers=2)

    for i, (clip1, clip2, label) in enumerate(train_loader):
        print('i: {}'.format(i))
        print('clip1.size:{}'.format(clip1.shape))
        print('clip2.size:{}'.format(clip2.shape))
        print(label)
        if label == -1:
            print('lable is not the same!! ERROR!')
            break
        print('-------------------------------------------------')

        if i == 5:
            break

    '''


    for i in range(10):
        index = random.randint(0, 9000)
        print('index:', index)
        coviar_data = CoviarData(params['mpeg_data'],
                                 'ucf101',
                                 params['mpeg_video_list'],
                                 'residual', get_augmentation(), 16, True, True).__getitem__(index)
        raw_data = ClassifyDataSet("/data2/video_data/UCF-101", mode='train').__getitem__(index)
        print('coviar_data:', coviar_data[0].shape, 'label:', coviar_data[1])
        print('raw_data:', raw_data[0].shape, 'label:', raw_data[1])
        print('-----------------------------------------------------------------')
    '''







