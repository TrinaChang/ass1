"""
   frac.py
   COMP9444, CSE, UNSW
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Full2Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full2Net, self).__init__()
        self.hidlayer1 = nn.Linear(2, hid)
        self.hidlayer2 = nn.Linear(hid, hid)
        self.outlayer = nn.Linear(hid,  1)
    def forward(self, input):
        self.hid1 = torch.tanh(self.hidlayer1(input))
        self.hid2 = torch.tanh(self.hidlayer2(self.hid1))
        out = torch.sigmoid(self.outlayer(self.hid2))
        return out

class Full3Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full3Net, self).__init__()
        self.hidlayer1 = nn.Linear(2, hid)
        self.hidlayer2 = nn.Linear(hid, hid)
        self.hidlayer3 = nn.Linear(hid, hid)
        self.outlayer = nn.Linear(hid,  1)        

    def forward(self, input):
        self.hid1 = torch.tanh(self.hidlayer1(input))
        self.hid2 = torch.tanh(self.hidlayer2(self.hid1))
        self.hid3 = torch.tanh(self.hidlayer2(self.hid2))
        out = torch.sigmoid(self.outlayer(self.hid3))
        return out

class DenseNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(DenseNet, self).__init__()
        self.layer1 = nn.Linear(2, num_hid)
        self.layer2 = nn.Linear(num_hid + 2, num_hid)
        self.layer3 = nn.Linear(2 + 2 * num_hid, 1)
    def forward(self, input):
        self.hid1 = torch.tanh(self.layer1(input))
        self.hid2 = torch.tanh(self.layer2(torch.cat((self.hid1, input), 1)))
        return torch.sigmoid(self.layer3(torch.cat((self.hid2, self.hid1, input), 1)))