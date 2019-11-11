"""Model Definition"""

import torch
import torch.nn as nn
from net_part import *


class C3D(nn.Module):
    def __init__(self, with_classifier=False, num_classes=101, return_features=False):
        super(C3D, self).__init__()
        self.with_classifier = with_classifier
        self.num_classes = num_classes
        self.return_feature = return_features
        self.conv1 = conv3d(3, 64);
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = conv3d(64, 128);
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3_1 = conv3d(128, 256);
        self.conv3_2 = conv3d(256, 256);
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4_1 = conv3d(256, 512);
        self.conv4_2 = conv3d(512, 512);
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5_1 = conv3d(512, 512);
        self.conv5_2 = conv3d(512, 512);
        self.pool5 = nn.AdaptiveAvgPool3d(1)
        if self.return_feature:
            self.feature_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # 9216
        if self.with_classifier:
            self.linear = nn.Linear(512, self.num_classes)


    def subforward(self, x):
        x = self.conv1(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        return x

    def forward(self, x):
        x = self.subforward(x)
        if self.return_feature == True:
            x = self.feature_pool(x)
            return x.view(x.shape[0], -1)
        x = self.pool5(x)
        out = x.view(-1, 512)

        if self.with_classifier == True:
            out = self.linear(out)
        # print(x1.shape)

        return out

if __name__ == '__main__':
    input_tensor = torch.autograd.Variable(torch.rand(1, 3, 16, 112, 112).cuda())

    c3d = C3D(return_features=True)
    c3d = c3d.cuda()
    output = c3d(input_tensor)

    print(output.shape)

