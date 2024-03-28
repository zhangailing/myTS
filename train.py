from dgl.data import CoraGraphDataset
import torch.nn.functional as F
import torch.optim as optim
import torch
import dgl

# 加载数据集
dataset = CoraGraphDataset()
g = dataset[0]

# 结构教师模型和特征教师模型
structure_teacher = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)
feature_teacher = MLP(g.ndata['feat'].shape[1], 16, dataset.num_classes)

# 优化器
structure_optimizer = optim.Adam(structure_teacher.parameters(), lr=0.01)
feature_optimizer = optim.Adam(feature_teacher.parameters(), lr=0.01)

for epoch in range(100):
    # 结构教师训练
    structure_teacher.train()
    logits = structure_teacher(g, g.ndata['feat'])
    loss = F.cross_entropy(logits[g.ndata['train_mask']], g.ndata['label'][g.ndata['train_mask']])
    
    structure_optimizer.zero_grad()
    loss.backward()
    structure_optimizer.step()
    
    # 特征教师训练
    feature_teacher.train()
    feature_logits = feature_teacher(g.ndata['feat'])
    feature_loss = F.cross_entropy(feature_logits[g.ndata['train_mask']], g.ndata['label'][g.ndata['train_mask']])
    
    feature_optimizer.zero_grad()
    feature_loss.backward()
    feature_optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Structure Loss: {loss.item()}, Feature Loss: {feature_loss.item()}')