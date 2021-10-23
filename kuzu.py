"""
   kuzu.py
   COMP9444, CSE, UNSW
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.lin = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return F.log_softmax(self.lin(x.view(x.size(0), -1)), dim=1)

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        self.in_to_hid = nn.Linear(28 * 28, 250)
        self.hid_to_out = nn.Linear(250, 10)

    def forward(self, x):
        x = torch.tanh(self.in_to_hid(x.view(x.size(0), -1)))
        return F.log_softmax(self.hid_to_out(x), dim=1)

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5)
        self.conv2 =  nn.Conv2d(64, 32, 5)
        self.lin1 = nn.Linear(512, 256)
        self.lin2 = nn.Linear(256, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.lin1(out))
        out = F.log_softmax(self.lin2(out), dim=1)
        return out 
