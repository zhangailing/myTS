import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import CoraGraphDataset
from dgl.nn import GraphConv

# 结构教师
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

# 特征教师
class MLP(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_feats, h_feats)
        self.fc2 = nn.Linear(h_feats, num_classes)
    
    def forward(self, in_feat):
        h = F.relu(self.fc1(in_feat))
        h = self.fc2(h)
        return h
