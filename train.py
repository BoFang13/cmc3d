from config import params
from model import c3d
from torch import nn, optim
import os
from datasets.raw_data import RawData
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import random
import numpy as np
from tensorboardX import SummaryWriter

params['data'] = 'UCF-101'
params['dataset'] ='/data2/video_data/UCF-101'

params['epoch_num'] = 300
params['batch_size'] = 8
params['num_workers'] = 4
params['learning_rate'] = 0.001

save_path = params['save_path_base'] + 'ft_classify_' + params['data']
gpu = 3

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


class AverageMeter(object):
    """   """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(train_loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()

    for step, (input, label) in enumerate(train_loader):
        data_time.update(time.time() - end)

        label = label.cuda()
        input = input .cuda()

        output = model(input)

        loss = criterion(output, label)
        prec1, prec5 = accuracy(output.data, label, topk=(1, 5))


        losses.update(loss.item(), input.size(0))

        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if (step + 1) % params['display'] == 0:
            print('------------------------------------------------')
            for param in optimizer.param_groups:
                print("lr:", params['lr'])

            p_str = "Epoch:[{0}][{1}/{2}]".format(epoch, step + 1, len(train_loader))
            print(p_str)

            p_str = "data_time:{data_time:.3f},batch_time:{batch_time:.3f}".format(data_time=data_time.val,
                                                                                   batch_time=batch_time.val)
            print(p_str)

            p_str = "loss:{loss:.5f}".format(loss=losses.avg)
            print(p_str)

            total_step = (epoch - 1) * len(train_loader) + step + 1
            writer.add_scalar('train/loss', losses.avg, total_step)
            writer.add_scalar('train/acc', top1.avg, total_step)

            p_str = "Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}".format(
                top1_acc=top1.avg,
                top5_acc=top5.avg
            )
            print(p_str)


def validation(val_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    end = time.time()
    total_loss = 0.0

    with torch.no_grad():
        for step, (inputs, labels) in enumerate(val_loader):
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.update(loss.item(), inputs.size(0))

            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))

            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            total_loss += loss.item()

            if (step + 1) % params['display'] == 0:
                print('-----------------------------validation-------------------')
                p_str = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(val_loader))
                print(p_str)

                p_str = 'data_time:{data_time:.3f},batch time:{batch_time:.3f}'.format(data_time=data_time.val,
                                                                                       batch_time=batch_time.val);
                print(p_str)

                p_str = 'loss:{loss:.5f}'.format(loss=losses.avg);
                print(p_str)

                p_str = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                    top1_acc=top1.avg,
                    top5_acc=top5.avg)
                print(p_str)

    avg_loss = total_loss / len(val_loader)
    return avg_loss, top1.avg


def load_pretrained_weights(ckpt_path):
    adjusted_weights = {}
    pretrained_weights =torch.load(ckpt_path, map_location='cpu')
    for name, params in pretrained_weights.items():
        print(name)
        if "module.base_network" in name:
            name = name[name.find('.') + 14:]
            adjusted_weights[name] = params

    return adjusted_weights


def loadcontinue_weights(path):
    adjusted_weights = {}
    pretrained_weights = torch.load(path, map_location='cpu')
    for name, params in pretrained_weights.items():
        print(name)

        if "module" in name and "linear" not in name:
            name = name[name.find('.') + 1:]
            adjusted_weights[name] = params

    return adjusted_weights


def main():
    model = c3d.C3D(with_classifier=True, num_classes=101)


if __name__ == '__main__':
    main()
