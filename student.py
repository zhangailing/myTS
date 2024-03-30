import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv

class Student(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(Student, self).__init__()
        # For graph structure-based learning
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)
        # For feature-based learning
        self.fc1 = nn.Linear(in_feats, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, g, features, use_structure=True):
        if use_structure:
            h = F.relu(self.conv1(g, features))
            h = self.conv2(g, h)
        else:
            h = F.relu(self.fc1(features))
            h = self.fc2(h)
        return h