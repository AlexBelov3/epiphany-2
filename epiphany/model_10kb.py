import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import sys
from utils import *


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
class old_Net(nn.Module):
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


class Net(nn.Module):
    def __init__(self, num_layers=1, input_channels=5, window_size=14000):

        super(Net, self).__init__()
        self.input_channels = input_channels
        self.window_size = window_size

        self.conv1 = ConvBlock(in_channels=self.input_channels, out_channels=70, kernel_width=17, stride=1, pool_size=4) #pool_size=4
        self.do1 = nn.Dropout(p=.1)
        self.conv2 = ConvBlock(in_channels=70, out_channels=90, kernel_width=7, stride=1, pool_size=4) #pool_size=4
        self.do2 = nn.Dropout(p=.1)
        self.conv3 = ConvBlock(in_channels=90, out_channels=70, kernel_width=5, stride=1, pool_size=4) #pool_size=4
        self.do3 = nn.Dropout(p=.1)
        self.conv4 = ConvBlock(in_channels=70, out_channels=50, kernel_width=5, stride=1)
        self.do4 = nn.Dropout(p=.1)
        self.conv5 = ConvBlock(in_channels=50, out_channels=20, kernel_width=5, stride=1)
        self.pool = nn.AdaptiveMaxPool1d(900 // 20)
        self.do5 = nn.Dropout(p=.1)

        self.rnn1 = nn.LSTM(input_size=900, hidden_size=1200, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.rnn2 = nn.LSTM(input_size=2400, hidden_size=1200, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.rnn3 = nn.LSTM(input_size=2400, hidden_size=1200, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2400, 900)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(900, 100) #200
        self.act2 = nn.ReLU()
        # # #ADDED:
        # self.fc3 = nn.Linear(100, 1)
        # self.act3 = nn.ReLU()
        # self.conv6 = ConvBlock(in_channels=20, out_channels=15, kernel_width=5, stride=1, pool_size=0)  # pool_size=4
        # self.do6 = nn.Dropout(p=.1)
        # self.conv7 = ConvBlock(in_channels=15, out_channels=10, kernel_width=5, stride=1, pool_size=0)  # pool_size=4
        # self.do7 = nn.Dropout(p=.1)
        # self.conv7 = ConvBlock(in_channels=10, out_channels=5, kernel_width=5, stride=1, pool_size=0)  # pool_size=4
        # self.do7 = nn.Dropout(p=.1)
        # self.conv8 = ConvBlock(in_channels=5, out_channels=1, kernel_width=5, stride=1, pool_size=0)  # pool_size=4
        # self.do8 = nn.Dropout(p=.1)

    def forward(self, x, hidden_state=None, seq_length=200):
        x = x.squeeze()
        assert x.shape[0] == self.input_channels, f"Expected {self.input_channels} input channels, but got {x.shape[0]}"
        x = torch.as_strided(x, (seq_length, self.input_channels, self.window_size), (100, x.shape[1], 1)) #stride = 1?
        # x = x.unsqueeze(0)
        # print(f"x input shape: {x.shape}")
        x = self.conv1(x)
        x = self.do1(x)
        x = self.conv2(x)
        x = self.do2(x)
        x = self.conv3(x)
        x = self.do3(x)
        x = self.conv4(x)
        x = self.do4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = self.do5(x)
        # print(f"x conv output shape: {x.shape}")
        x = x.view(1, seq_length, x.shape[1] * x.shape[2])
        res1, hidden_state = self.rnn1(x, None)
        res2, hidden_state = self.rnn2(res1, None)
        res2 = res2 + res1
        res3, hidden_state = self.rnn3(res2, None)
        # print(f"x LSTM output shape: {res2.shape}")
        x = self.fc(res2 + res3)
        x = self.act(x)
        x = self.fc2(x)
        # ADDED LINES:
        # x = self.act2(x)
        # x = self.fc3(x)
        # x = self.conv6(x)
        # x = self.do6(x)
        # x = self.conv7(x)
        # x = self.do7(x)
        # x = self.conv8(x)
        # x = self.do8(x)
        # print(f"x output shape: {x.shape}")
        x = x.squeeze()
        # # print(f"x output shape: {x.shape}")
        x_R, x_L = extract_diagonals(x)
        x = torch.cat((x_R, x_L), 0)
        x = x.unsqueeze(0)
        return x#, hidden_state

    def loss(self, prediction, label, seq_length=200, reduction='mean', lam=1):
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

class Net2(nn.Module):
    def __init__(self, num_layers=1, input_channels=5, window_size=14000):

        super(Net2, self).__init__()
        self.input_channels = input_channels
        self.window_size = window_size

        self.cov_extractor = nn.Sequential(
            nn.Conv1d(
                in_channels=5, out_channels=16, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding=1,
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            resblock(16),
            nn.MaxPool1d(kernel_size=2),
            resblock(16),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding=1,
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding=1,
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=16,
                out_channels=1,  # 16
                kernel_size=3,
                stride=1,
                dilation=1,
                padding=1,
            ),
            # nn.BatchNorm1d(16), #1
            nn.ReLU(),
        )
        self.rnn1 = nn.LSTM(input_size=343, hidden_size=1200, num_layers=num_layers, batch_first=True, #no res blocks: in=1375
                            bidirectional=True)
        self.rnn2 = nn.LSTM(input_size=2400, hidden_size=1200, num_layers=num_layers, batch_first=True,
                            bidirectional=True)
        self.rnn3 = nn.LSTM(input_size=2400, hidden_size=1200, num_layers=num_layers, batch_first=True,
                            bidirectional=True)
        # self.rnn1 = nn.LSTM(input_size=343, hidden_size=600, num_layers=num_layers, batch_first=True, bidirectional=True)
        # self.rnn2 = nn.LSTM(input_size=1200, hidden_size=600, num_layers=num_layers, batch_first=True,
        #                     bidirectional=True)
        # self.rnn3 = nn.LSTM(input_size=1200, hidden_size=600, num_layers=num_layers, batch_first=True,
        #                     bidirectional=True)
        self.fc = nn.Linear(2400, 900)
        # self.act = nn.ReLU()
        self.fc2 = nn.Linear(900, 200)
        # self.act2 = nn.ReLU()

    def forward(self, x, hidden_state=None, seq_length=200):
        x = x.squeeze()
        assert x.shape[0] == self.input_channels, f"Expected {self.input_channels} input channels, but got {x.shape[0]}"
        x = x.unsqueeze(0)
        if x.ndimension() == 2:
            x = x.unsqueeze(0)
        x = self.cov_extractor(x)
        x = torch.flatten(x, 1)
        # x = x.view(1, seq_length, x.shape[1] * x.shape[2])
        res1, hidden_state = self.rnn1(x, None)
        res2, hidden_state = self.rnn2(res1, None)
        res2 = res2 + res1
        res3, hidden_state = self.rnn3(res2, None)
        x = self.fc(res2 + res3)
        # x = self.act(x)
        x = self.fc2(x)
        # x = self.act2(x)
        return x

    def loss(self, prediction, label, seq_length=200, reduction='mean', lam=1):
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
# class Net(nn.Module):
#     def __init__(self, num_layers=1, input_channels=5, window_size=14000):
#         super(Net, self).__init__()
#
#         self.cov_extractor = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=5, out_channels=16, kernel_size=5, stride=1, padding=2
#             ),
#             # nn.BatchNorm1d(34000),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2),
#             nn.Conv1d(
#                 in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2
#             ),
#             # nn.BatchNorm1d(34000),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2),
#             nn.Conv1d(
#                 in_channels=16,
#                 out_channels=16,
#                 kernel_size=3,
#                 stride=1,
#                 dilation=1,
#                 padding=1,
#             ),
#             # nn.BatchNorm1d(34000),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2),
#             # resblock(34000),
#             nn.MaxPool1d(kernel_size=2),
#             # resblock(34000),
#             nn.MaxPool1d(kernel_size=2),
#             nn.Conv1d(
#                 in_channels=16,
#                 out_channels=16,
#                 kernel_size=3,
#                 stride=1,
#                 dilation=1,
#                 padding=1,
#             ),
#             # nn.BatchNorm1d(34000),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2),
#             nn.Conv1d(
#                 in_channels=16,
#                 out_channels=16,
#                 kernel_size=3,
#                 stride=1,
#                 dilation=1,
#                 padding=1,
#             ),
#             # nn.BatchNorm1d(34000),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2),
#             nn.Conv1d(
#                 in_channels=16,
#                 out_channels=16,
#                 kernel_size=3,
#                 stride=1,
#                 dilation=1,
#                 padding=1,
#             ),
#             # nn.BatchNorm1d(34000),
#             nn.ReLU(),
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=(265), out_features=512),  # 992
#         )
#
#     def forward(self, x):
#         x = self.cov_extractor(x)
#         x = torch.flatten(x, 1)
#         x_out = self.classifier(x)
#
#         return x_out
#
#     def loss(self, prediction, label, seq_length=200, reduction='mean', lam=1):
#         l1_loss = 0
#         if isinstance(prediction, np.ndarray):
#             prediction = torch.tensor(prediction)
#         if isinstance(label, np.ndarray):
#             label = torch.tensor(label)
#
#         if prediction.ndim != 1 or label.ndim != 1:
#             prediction = prediction.view(-1)
#             label = label.view(-1)
#
#         if prediction.size() != label.size():
#             raise ValueError(
#                 f"Shape mismatch: prediction size {prediction.size()} does not match label size {label.size()}")
#
#         # Compute L1 and L2 losses
#         # l1_loss = F.l1_loss(prediction, label, reduction=reduction)
#         l2_loss = F.mse_loss(prediction, label, reduction=reduction)
#
#         # Combine losses with lambda
#         total_loss = lam * l2_loss + (1 - lam) * l1_loss
#         return total_loss


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


class branch_outerprod(nn.Module):
    def __init__(self):
        super(branch_outerprod, self).__init__()

        # self.total_extractor_2d = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2),
        #     # nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2),
        #     # nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2),
        #     # nn.BatchNorm2d(16),
        #     nn.ReLU(),
        # )

        self.total_extractor_2d = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
            # nn.BatchNorm2d(16),
            nn.ReLU(),

        )

        # Calculate the final output size after all convolutions and pooling
        # Input size: 1700x1700
        # After first conv+pool: (1700-3)/2 + 1 = 849/2 = 424.5, rounded down to 424x424
        # After second conv+pool: (424-3)/2 + 1 = 211/2 = 105.5, rounded down to 105x105
        # After third conv+pool: (105-3)/2 + 1 = 51/2 = 25.5, rounded down to 25x25

        # So the final output feature map size is 16 (channels) * 25 * 25
        final_feature_map_size = 144 #2704

        self.classifier = nn.Sequential(
            nn.Linear(in_features=final_feature_map_size, out_features=512),
        )
        self.classifier2 = nn.Sequential(nn.Linear(in_features=512, out_features=200))

    def forward(self, x2):
        # # x2 = F.softmax(x2, dim=1)
        # Apply softmax across the height and width dimensions
        batch_size, height, width = x2.size()
        x2 = x2.view(batch_size, -1)  # Flatten height and width dimensions
        x2 = F.softmax(x2, dim=1)  # Apply softmax
        x2 = x2.view(batch_size, height, width)  # Reshape back to original dimensions

        x3 = self.total_extractor_2d(x2)
        x3 = torch.flatten(x3, 1)
        x3 = self.classifier(x3)
        x4 = self.classifier2(x3)
        return x4


class trunk(nn.Module):
    def __init__(self, right, left):
        super(trunk, self).__init__()

        self.right = right
        self.left = left

        self.out = nn.Sequential(
            # nn.Linear(in_features=(916), out_features=512), #replace 116 with 512 * 2
            nn.Linear(in_features=(400), out_features=200),
            # nn.Linear(in_features=(512), out_features=100),
        )

    def forward(self, x):
        x_copy = x.clone()
        x = self.left(x)
        x2 = self.right(x_copy)
        x = x.reshape(1, 200)
        x2 = x2.reshape(1, 200)
        x = self.out(torch.cat((x, x2), 1)) # x = self.out(torch.cat((x, torch.t(x2)), 1))
        return x

    def loss(self, prediction, label, seq_length = 200, reduction='mean', lam=1):
        l1_loss = 0
        if isinstance(prediction, np.ndarray):
            prediction = torch.tensor(prediction)
        if isinstance(label, np.ndarray):
            label = torch.tensor(label)
        if prediction.ndim != 1 or label.ndim != 1:
            prediction = prediction.view(-1)
            label = label.view(-1)
        # Compute L1 and L2 losses
        # l1_loss = F.l1_loss(prediction, label, reduction=reduction)
        l2_loss = F.mse_loss(prediction, label, reduction=reduction)

        # Combine losses with lambda
        total_loss = lam * l2_loss + (1 - lam) * l1_loss
        return total_loss


# ------------------------------------------------------------------------------------------------------------------
#CHROMAFOLD:
class resblock(nn.Module):
    def __init__(self, ni):
        super(resblock, self).__init__()
        self.blocks = nn.Sequential(
            nn.Conv1d(ni, ni, 3, 1, 1),
            nn.BatchNorm1d(ni),
            nn.ReLU(),
            nn.Conv1d(ni, ni, 3, 1, 1),
            nn.BatchNorm1d(ni),
            nn.ReLU(),
        )

    def forward(self, x):
        residual = x
        out = self.blocks(x)
        out = out + residual

        return out


class resblock_2d(nn.Module): # COME BACK TO THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __init__(self, ni):
        super(resblock_2d, self).__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(ni, ni, 3, 1, 1),
            nn.BatchNorm2d(ni),
            nn.ReLU(),
            nn.Conv2d(ni, ni, 3, 1, 1),
            nn.BatchNorm2d(ni),
            nn.ReLU(),
        )

    def forward(self, x):
        residual = x
        out = self.blocks(x)
        out, residual = out.squeeze(), residual.squeeze()
        out = out + residual
        out = out.unsqueeze(0)

        return out


class symmetrize_bulk(nn.Module):
    def __init__(self):
        super(symmetrize_bulk, self).__init__()

    def forward(self, x):
        if len(x.shape) == 2:
            print("not implemented")
            return None
        else:
            if len(x.shape) == 3:
                a, b, c = x.shape
                x = x.reshape(a, b, 1, c)
                x = x.repeat(1, 1, c, 1)
                x_t = x.permute(0, 1, 3, 2)
                x_sym = torch.concat((x, x_t), axis=1)  # (x+x_t)/2
                return x_sym
            else:
                return None


class outter_prod(nn.Module):
    def __init__(self):
        super(outter_prod, self).__init__()

    def forward(self, x):
        if len(x.shape) == 2:
            print("not implemented")
            return None
        else:
            if len(x.shape) == 3:
                x = x.squeeze()
                binned_signals = []
                for i in range(np.shape(x)[0]):
                    binned_signals.append(np.outer(x[i].cpu(),x[i].cpu()))
                co_signal = torch.tensor(binned_signals)
                a, b, c = co_signal.shape
                co_signal = co_signal.reshape(1, a, b, c)
                return torch.tensor(co_signal).cuda()
            else:
                return None

class outter_prod_big(nn.Module):
    def __init__(self):
        super(outter_prod_big, self).__init__()

    def forward(self, x):
        if len(x.shape) == 2:
            print("not implemented")
            return None
        else:
            if len(x.shape) == 3:
                x = x.squeeze()
                binned_signals = []
                for i in range(np.shape(x)[0]):
                    binned_signals.append(x[i].cpu())
                binned_signals = np.outer(binned_signals, binned_signals)
                co_signal = torch.tensor(binned_signals)
                a, b, c = co_signal.shape
                co_signal = co_signal.reshape(1, a, b, c)
                return torch.tensor(co_signal).cuda()
            else:
                return None

class branch_outter_prod_small(nn.Module):
    def __init__(self):
        super(branch_outter_prod_small, self).__init__()
        pbulk_res = 50

        self.bulk_summed_2d = nn.Sequential(
            nn.AvgPool1d(kernel_size=np.int64(1e04 / pbulk_res)), outter_prod()
        )

        self.total_extractor_2d = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=64, kernel_size=3, stride=2),
            # nn.Conv2d(in_channels=36, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # self.classifier = nn.Sequential(
        #     nn.Linear(in_features=(400), out_features=512), #nn.Linear(in_features=(1936), out_features=512),
        # )
        self.classifier2 = nn.Sequential(
            nn.Linear(in_features=(576), out_features=200))  # in = 400 for window_size=20,000

    def forward(self, x2):
        x3_2d = self.bulk_summed_2d(x2)
        x4 = self.total_extractor_2d(x3_2d)
        x4 = torch.flatten(x4, 1)
        x4 = self.classifier2(x4)
        # print(f"x4 shape: {x4.shape}")
        return x4
#
class branch_outter_prod_big(nn.Module):
    def __init__(self):
        super(branch_outter_prod_big, self).__init__()
        pbulk_res = 50

        self.bulk_summed_2d = nn.Sequential(
            nn.AvgPool1d(kernel_size=np.int64(1e04 / pbulk_res)), outter_prod_big()
        )

        self.total_extractor_2d = nn.Sequential(
            nn.Conv2d(in_channels=37, out_channels=64, kernel_size=3, stride=2),
            # nn.Conv2d(in_channels=36, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # self.classifier = nn.Sequential(
        #     nn.Linear(in_features=(400), out_features=512), #nn.Linear(in_features=(1936), out_features=512),
        # )
        self.classifier2 = nn.Sequential(
            nn.Linear(in_features=(400), out_features=200))  # (nn.Linear(in_features=(512), out_features=200))

    def forward(self, x2):
        x3_2d = self.bulk_summed_2d(x2)
        x4 = self.total_extractor_2d(x3_2d)
        x4 = torch.flatten(x4, 1)
        # x4 = self.classifier2(x4)
        print(f"x4 shape: {x4.shape}")
        return x4

class branch_pbulk(nn.Module):
    def __init__(self):
        super(branch_pbulk, self).__init__()

        pbulk_res = 50
        scatac_res = 500

        # self.bulk_summed_2d = nn.Sequential(
        #     nn.AvgPool1d(kernel_size=np.int64(1e04 / pbulk_res)), symmetrize_bulk()
        # )

        self.bulk_summed_2d = nn.Sequential(
            nn.AvgPool1d(kernel_size=np.int64(1e04 / pbulk_res)), outter_prod()
        )

        self.bulk_extractor_2d = nn.Sequential(
            nn.Conv1d(
                in_channels=5,
                out_channels=16,
                kernel_size=11,
                stride=1,
                dilation=1,
                padding="same",
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=7,
                stride=1,
                dilation=1,
                padding="same",
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                stride=1,
                dilation=1,
                padding="same",
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                stride=1,
                dilation=1,
                padding="same",
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                stride=1,
                dilation=1,
                padding="same",
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                stride=1,
                dilation=2,
                padding="same",
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                stride=1,
                dilation=3,
                padding="same",
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                stride=1,
                dilation=5,
                padding="same",
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                stride=1,
                dilation=5,
                padding="same",
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                stride=1,
                dilation=7,
                padding="same",
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                stride=1,
                dilation=11,
                padding="same",
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                stride=1,
                dilation=11,
                padding="same",
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5),
            nn.Conv1d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding="same",
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5),
            nn.Conv1d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding="same",
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding="same",
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            symmetrize_bulk(),
        )

        self.total_extractor_2d = nn.Sequential(
            nn.Conv2d(in_channels=37, out_channels=64, kernel_size=3, stride=2), #nn.Conv2d(in_channels=36, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # self.classifier = nn.Sequential(
        #     nn.Linear(in_features=(400), out_features=512), #nn.Linear(in_features=(1936), out_features=512),
        # )
        self.classifier2 = nn.Sequential(nn.Linear(in_features=(400), out_features=200)) #(nn.Linear(in_features=(512), out_features=200))

    def forward(self, x2):
        x3_2d = self.bulk_summed_2d(x2)
        x2_2d = self.bulk_extractor_2d(x2)
        x4 = torch.cat((x3_2d, x2_2d), 1)
        x4 = self.total_extractor_2d(x4)
        x4 = torch.flatten(x4, 1)
        x4 = self.classifier2(x4)

        return x4

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


class branch_cov(nn.Module):
    def __init__(self):
        super(branch_cov, self).__init__()

        self.cov_extractor = nn.Sequential(
            nn.Conv1d(
                in_channels=5, out_channels=16, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # PrintLayer(),
            # FirstConvLayer(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding=1,
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            resblock(16),
            nn.MaxPool1d(kernel_size=2),
            resblock(16),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding=1,
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding=1,
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=16,
                out_channels=1, #16
                kernel_size=3,
                stride=1,
                dilation=1,
                padding=1,
            ),
            # nn.BatchNorm1d(16), #1
            nn.ReLU(),
        )


        self.classifier = nn.Sequential(
            # nn.Linear(in_features=(265), out_features=512), #992
            nn.Linear(in_features=343, out_features=200), #input = 312 for window_size == 20,000
        )


    def forward(self, x):
        # try:
        #     x = self.cov_extractor(x)
        # except:
        #     x = self.cov_extractor_backup(x)
        if x.ndimension() == 2:
            x = x.unsqueeze(0)
        x = self.cov_extractor(x)
        x = torch.flatten(x, 1)
        x_out = self.classifier(x)

        return x_out

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


class branch_cov_2d(nn.Module):
    def __init__(self):
        super(branch_cov_2d, self).__init__()

        self.cov_extractor = nn.Sequential(
            # PrintLayer(),
            # FirstConvLayer(),
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=(1,5), stride=1, padding=2
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=16, out_channels=16, kernel_size=(1,5), stride=1, padding=2
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(1,3),
                stride=1,
                dilation=1,
                padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            resblock_2d(16),
            # --------------------------------------------------
            # nn.Conv2d(16, 16, (1, 3), 1, 1),
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
            # nn.Conv2d(16, 16, (1, 3), 1, 1),
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            resblock_2d(16),
            # --------------------------------------------------
            # nn.Conv2d(16, 16, (1, 3), 1, 1),
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
            # nn.Conv2d(16, 16, (1, 3), 1, 1),
            # nn.BatchNorm2d(16),
            # nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(1,3),
                stride=1,
                dilation=1,
                padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(1,3),
                stride=1,
                dilation=1,
                padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=16,
                out_channels=1, #16
                kernel_size=(1,3),
                stride=1,
                dilation=1,
                padding=1,
            ),
            # nn.BatchNorm2d(16), #1
            nn.ReLU(),
        )


        self.classifier = nn.Sequential(
            nn.Linear(in_features=1875, out_features=512),
            nn.Linear(in_features=512, out_features=200),
        )


    def forward(self, x):
        # try:
        #     x = self.cov_extractor(x)
        # except:
        #     x = self.cov_extractor_backup(x)
        # x = x.view(x.size(0), 1, 80, 400)
        # try:
        #     x = x.view(x.size(0), 1, 80, 2500)
        # except:
        #     x = x.view(x.size(0), 1, 80, 2499)
        if x.ndimension() == 2:
            a, b = x.shape
            x = x.view(1, 1, a, b)
        elif x.ndimension() == 3:
            a, b, c = x.shape
            x = x.view(1, a, b, c)
        x = self.cov_extractor(x)
        x = torch.flatten(x, 1)
        x_out = self.classifier(x)

        return x_out

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
class branch_modified_pbulk(nn.Module):
    def __init__(self):
        super(branch_modified_pbulk, self).__init__()

        pbulk_res = 78 #50
        scatac_res = 500


        # self.bulk_summed_2d = nn.Sequential(
        #     nn.AvgPool1d(kernel_size=np.int64(1e04 / pbulk_res)), symmetrize_bulk()
        # )

        self.bulk_summed_2d = nn.Sequential(
            nn.AvgPool1d(kernel_size=np.int64(1e04 / pbulk_res)), outter_prod()
        )

        self.cov_extractor = nn.Sequential(
            nn.Conv1d(
                in_channels=5, out_channels=16, kernel_size=5, stride=1, padding=2, dilation=1 #padding="same"
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2, dilation=1
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding=1, #"same"
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            resblock(16),
            nn.MaxPool1d(kernel_size=2),
            resblock(16),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding=1, #"same"
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding=1, #"same"
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding=1, #"same"
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            symmetrize_bulk()
        )


        self.total_extractor_2d = nn.Sequential(
            nn.Conv2d(in_channels=37, out_channels=64, kernel_size=3, stride=2), #nn.Conv2d(in_channels=36, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=(1296), out_features=512), #nn.Linear(in_features=(1936), out_features=512),
        )
        self.classifier2 = nn.Sequential(nn.Linear(in_features=(512), out_features=200)) #(nn.Linear(in_features=(512), out_features=200))

    def forward(self, x):
        x3_2d = self.bulk_summed_2d(x)
        x2_2d = self.cov_extractor(x)
        # x2_2d = self.bulk_extractor_2d(x)
        # print(f"x shape: {x.shape}")
        # print(f"x3_2d shape: {x3_2d.shape}")
        # print(f"x2_2d shape: {x2_2d.shape}")

        x4 = torch.cat((x3_2d, x2_2d), 1)
        x4 = self.total_extractor_2d(x4)
        x4 = torch.flatten(x4, 1)
        x4 = self.classifier(x4)
        x4 = self.classifier2(x4)
        return x4

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


# class trunk(nn.Module):
#     def __init__(self, branch_pbulk, branch_cov):
#         super(trunk, self).__init__()
#
#         self.branch_pbulk = branch_pbulk
#         self.branch_cov = branch_cov
#
#         self.out = nn.Sequential(
#             nn.Linear(in_features=(512 * 2), out_features=512),
#             nn.Linear(in_features=(512), out_features=200),
#         )
#
#     def forward(self, x, x2):
#         x = self.branch_cov(x, x2)
#         with torch.no_grad():
#             x2 = self.branch_pbulk(x2)
#         x = self.out(torch.cat((x, x2), 1))
#
#         return x