import torch
import dgl
import torch.nn.functional as F
import torch.optim as optim

torch.autograd.set_detect_anomaly(True)

from torch.nn import CrossEntropyLoss
from dgl.data import CoraGraphDataset,CiteseerGraphDataset,PubmedGraphDataset
from teacher import GCN, MLP
from student import Student
from loss import distillation_loss
from utils import calculate_accuracy



# Load the dataset
dataset = CiteseerGraphDataset()
g = dataset[0]

# Initialize models
structure_teacher = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)
feature_teacher = MLP(g.ndata['feat'].shape[1], 16, dataset.num_classes)
student = Student(g.ndata['feat'].shape[1], 16, dataset.num_classes)

# Optimizers
structure_teacher_optimizer = optim.Adam(structure_teacher.parameters(), lr=0.01)
feature_teacher_optimizer = optim.Adam(feature_teacher.parameters(), lr=0.01)
student_optimizer = optim.Adam(student.parameters(), lr=0.01)

# Loss function
ce_loss = CrossEntropyLoss()

# Training loop
epochs = 100
temperature = 2.0  # For distillation loss


for epoch in range(epochs):
    # Forward pass for teachers
    structure_teacher_logits = structure_teacher(g, g.ndata['feat'])
    feature_teacher_logits = feature_teacher(g.ndata['feat'])

    # Calculate traditional loss for teachers
    structure_teacher_loss = ce_loss(structure_teacher_logits[g.ndata['train_mask']], g.ndata['label'][g.ndata['train_mask']])
    feature_teacher_loss = ce_loss(feature_teacher_logits[g.ndata['train_mask']], g.ndata['label'][g.ndata['train_mask']])

    # Backward and optimize for teachers
    structure_teacher_optimizer.zero_grad()
    structure_teacher_loss.backward()
    structure_teacher_optimizer.step()

    feature_teacher_optimizer.zero_grad()
    feature_teacher_loss.backward()
    feature_teacher_optimizer.step()

    # Train with structure teacher
    student_logits = student(g, g.ndata['feat'], use_structure=True)
    student_loss = distillation_loss(structure_teacher_logits, student_logits, temperature)
    student_optimizer.zero_grad()
    student_loss.backward(retain_graph=True)
    student_optimizer.step()

    # Train with feature teacher
    student_logits = student(g, g.ndata['feat'], use_structure=False)
    student_loss = distillation_loss(feature_teacher_logits, student_logits, temperature)
    student_optimizer.zero_grad()
    student_loss.backward(retain_graph=True)
    student_optimizer.step()

    # Calculate accuracy (assuming a function calculate_accuracy(logits, labels) is defined)
    structure_teacher_acc = calculate_accuracy(structure_teacher_logits[g.ndata['train_mask']], g.ndata['label'][g.ndata['train_mask']])
    feature_teacher_acc = calculate_accuracy(feature_teacher_logits[g.ndata['train_mask']], g.ndata['label'][g.ndata['train_mask']])
    student_acc = calculate_accuracy(student_logits[g.ndata['train_mask']], g.ndata['label'][g.ndata['train_mask']])

    if epoch % 10 == 0:
        print(f"Epoch {epoch}")
        print(f"Structure Teacher Loss: {structure_teacher_loss.item()}, Acc: {structure_teacher_acc}")
        print(f"Feature Teacher Loss: {feature_teacher_loss.item()}, Acc: {feature_teacher_acc}")
        print(f"Structure Student Loss: {student_loss.item()}, Acc: {student_acc}")
