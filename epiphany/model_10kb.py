import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import sys


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_width, stride=1, pool_size=0):

        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_width, stride=1)
        self.act = nn.ReLU()
        self.pool_size = pool_size

        if pool_size > 0:
            self.pool = nn.MaxPool1d(self.pool_size, self.pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)

        if self.pool_size > 0:
            x = self.pool(x)

        return x


# USE THIS MODEL
class Net(nn.Module):
    def __init__(self, num_layers=1, input_channels=5, window_size=14000):

        super(Net, self).__init__()
        self.input_channels = input_channels
        self.window_size = window_size

        self.conv1 = ConvBlock(in_channels=self.input_channels, out_channels=70, kernel_width=17, stride=1, pool_size=4)
        self.do1 = nn.Dropout(p = .1)
        self.conv2 = ConvBlock(in_channels=70, out_channels=90, kernel_width=7, stride=1, pool_size=4)
        self.do2 = nn.Dropout(p = .1)
        self.conv3 = ConvBlock(in_channels=90, out_channels=70, kernel_width=5, stride=1, pool_size=4)
        self.do3 = nn.Dropout(p = .1)
        self.conv4 = ConvBlock(in_channels=70, out_channels=20, kernel_width=5, stride=1)
        self.pool = nn.AdaptiveMaxPool1d(900//20)
        self.do4 = nn.Dropout(p = .1)
  
        self.rnn1 = nn.LSTM(input_size=900, hidden_size=1200, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.rnn2 = nn.LSTM(input_size=2400, hidden_size=1200, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.rnn3 = nn.LSTM(input_size=2400, hidden_size=1200, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2400, 900)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(900, 100)
        self.act2 = nn.ReLU()
        #ADDED:
        # self.fc3 = nn.Linear(100, 1)
        # self.act3 = nn.ReLU()

    def forward(self, x, hidden_state=None, seq_length=200):

        assert x.shape[0] == self.input_channels, f"Expected {self.input_channels} input channels, but got {x.shape[0]}"
        x = torch.as_strided(x, (seq_length, self.input_channels, self.window_size), (100, x.shape[1], 1))

        x = self.conv1(x)
        x = self.do1(x)
        x = self.conv2(x)
        x = self.do2(x)
        x = self.conv3(x)
        x = self.do3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.do4(x)

        x = x.view(1, seq_length, x.shape[1]*x.shape[2])  
        res1, hidden_state = self.rnn1(x, None)
        res2, hidden_state = self.rnn2(res1, None)
        res2 = res2 + res1
        res3, hidden_state = self.rnn3(res2, None)
        x = self.fc(res2 + res3)
        x = self.act(x)
        x = self.fc2(x)
        #ADDED LINES:
        # x = self.act2(x)
        # x = self.fc3(x)
        return x, hidden_state

    def loss(self, prediction, label, seq_length = 200, reduction='mean', lam=1):
        l1_loss = 0
        if isinstance(prediction, np.ndarray):
            prediction = torch.tensor(prediction)
        if isinstance(label, np.ndarray):
            label = torch.tensor(label)

        if prediction.ndim != 1 or label.ndim != 1:
            prediction = prediction.view(-1)
            label = label.view(-1)

        if prediction.size() != label.size():
            raise ValueError(
                f"Shape mismatch: prediction size {prediction.size()} does not match label size {label.size()}")

        # Compute L1 and L2 losses
        # l1_loss = F.l1_loss(prediction, label, reduction=reduction)
        l2_loss = F.mse_loss(prediction, label, reduction=reduction)

        # Combine losses with lambda
        total_loss = lam * l2_loss + (1 - lam) * l1_loss
        return total_loss


class Disc(nn.Module):
    
    def __init__(self):

        super(Disc, self).__init__()
        self.conv1 = nn.Conv2d(1, 15, 5, stride=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(15, 25, 5, stride=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(25, 25, 5, stride=1)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(25, 10, 5, stride=1)
        self.act4 = nn.ReLU()
        self.fc1 = nn.Linear(850, 1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.act3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.act4(x)
        x = x.view(1, -1)
        x = self.fc1(x)

        return x

    def loss(self, prediction, label, reduction='mean'):
        
        loss_val = F.binary_cross_entropy_with_logits(prediction, label, reduction='mean')
        return loss_val


# class branch_pbulk(nn.Module):
#     def __init__(self):
#         super(branch_pbulk, self).__init__()
#
#         self.total_extractor_2d = nn.Sequential(
#             nn.Conv2d(in_channels=36, out_channels=64, kernel_size=3, stride=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=(1936), out_features=512),
#         )
#         self.classifier2 = nn.Sequential(nn.Linear(in_features=(512), out_features=200))
#     def forward(self, x2):
#         x3 = self.total_extractor_2d(x2)
#         x3 = torch.flatten(x3, 1)
#         x3 = self.classifier(x3)
#         return x3


class branch_pbulk(nn.Module):
    def __init__(self):
        super(branch_pbulk, self).__init__()

        self.total_extractor_2d = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # Calculate the final output size after all convolutions and pooling
        # Input size: 1700x1700
        # After first conv+pool: (1700-3)/2 + 1 = 849/2 = 424.5, rounded down to 424x424
        # After second conv+pool: (424-3)/2 + 1 = 211/2 = 105.5, rounded down to 105x105
        # After third conv+pool: (105-3)/2 + 1 = 51/2 = 25.5, rounded down to 25x25

        # So the final output feature map size is 16 (channels) * 25 * 25
        final_feature_map_size = 2704

        self.classifier = nn.Sequential(
            nn.Linear(in_features=final_feature_map_size, out_features=512),
        )
        self.classifier2 = nn.Sequential(nn.Linear(in_features=512, out_features=200))

    def forward(self, x2):
        x3 = self.total_extractor_2d(x2)
        x3 = torch.flatten(x3, 1)
        x3 = self.classifier(x3)
        x4 = self.classifier2(x3)
        return x4


class trunk(nn.Module):
    def __init__(self, branch_pbulk, Net):
        super(trunk, self).__init__()

        self.branch_pbulk = branch_pbulk
        self.Net = Net

        self.out = nn.Sequential(
            nn.Linear(in_features=(512 * 2), out_features=512),
            nn.Linear(in_features=(512), out_features=200),
        )

    def forward(self, x, x2):
        x = self.Net(x)[0].squeeze()
        with torch.no_grad():
            x2 = self.branch_pbulk(x2)
        x = self.out(torch.cat((x, x2.transpose()), 1))

        return x









