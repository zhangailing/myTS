import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv

# 结构学生
class StructureStudent(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(StructureStudent, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)

    def forward(self, g, features):
        h = F.relu(self.conv1(g, features))
        h = self.conv2(g, h)
        return h
# 特征学生  
class FeatureStudent(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(FeatureStudent, self).__init__()
        self.fc1 = nn.Linear(in_feats, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, features):
        h = F.relu(self.fc1(features))
        h = self.fc2(h)
        return h

# 示例参数
in_feats = 10
hidden_size = 16
num_classes = 3

# 创建模型实例
model = StructureStudent(in_feats, hidden_size, num_classes)
model = FeatureStudent(in_feats, hidden_size, num_classes)
