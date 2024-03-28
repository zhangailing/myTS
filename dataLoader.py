import dgl
import torch
from dgl.data import CiteseerGraphDataset
from torch.utils.data import DataLoader
from dgl.dataloading import GraphDataLoader
from dgl.dataloading import NodeDataLoader

# 加载Citeseer数据集
dataset = CiteseerGraphDataset()
g = dataset[0]

# 为节点分类准备掩码
g.ndata['features'] = g.ndata['feat']
g.ndata['labels'] = g.ndata['label']
g.ndata['train_mask'] = g.ndata['train_mask']

# 使用DGL的NodeDataLoader来创建训练数据加载器
train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
train_loader = NodeDataLoader(
    g,
    train_nid,
    sampler=dgl.dataloading.MultiLayerFullNeighborSampler(2),  # 假设我们使用2层邻居采样
    batch_size=64,
    shuffle=True,
    drop_last=False,
)

# 示例：遍历train_loader
for step, (input_nodes, output_nodes, blocks) in enumerate(train_loader):
    # input_nodes是当前batch所有需要的节点id
    # output_nodes是当前batch的输出节点id
    # blocks是一个包含DGLBlock对象的列表，它们表示消息传递的计算图
    batch_inputs = blocks[0].srcdata['features']  # 输入特征
    batch_labels = blocks[-1].dstdata['labels']   # 输出节点的标签
    # 你的模型训练代码...
    break  # 仅为示例，实际训练时应去掉这个break
