import torch
import dgl
import torch.nn.functional as F
import torch.optim as optim

torch.autograd.set_detect_anomaly(True)

from torch.nn import CrossEntropyLoss
from dgl.data import CoraGraphDataset
from teacher import GCN, MLP
from student import StructureStudent, FeatureStudent
from loss import distillation_loss
from utils import calculate_accuracy



# Load the dataset
dataset = CoraGraphDataset()
g = dataset[0]

# Initialize models
structure_teacher = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)
feature_teacher = MLP(g.ndata['feat'].shape[1], 16, dataset.num_classes)
structure_student = StructureStudent(g.ndata['feat'].shape[1], 16, dataset.num_classes)
feature_student = FeatureStudent(g.ndata['feat'].shape[1], 16, dataset.num_classes)


# Optimizers
structure_teacher_optimizer = optim.Adam(structure_teacher.parameters(), lr=0.01)
feature_teacher_optimizer = optim.Adam(feature_teacher.parameters(), lr=0.01)
structure_student_optimizer = optim.Adam(structure_student.parameters(), lr=0.01)
feature_student_optimizer = optim.Adam(feature_student.parameters(), lr=0.01)

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

    # Forward pass for students
    structure_student_logits = structure_student(g, g.ndata['feat'])
    feature_student_logits = feature_student(g.ndata['feat'])

    # Calculate distillation loss for students
    structure_student_distillation_loss = distillation_loss(structure_teacher_logits, structure_student_logits, temperature)
    feature_student_distillation_loss = distillation_loss(feature_teacher_logits, feature_student_logits, temperature)

    # Backward and optimize for teachers
    structure_teacher_optimizer.zero_grad()
    structure_teacher_loss.backward(retain_graph=True)
    structure_teacher_optimizer.step()

    feature_teacher_optimizer.zero_grad()
    feature_teacher_loss.backward(retain_graph=True)
    feature_teacher_optimizer.step()

    # Backward and optimize for students
    structure_student_optimizer.zero_grad()
    structure_student_distillation_loss.backward(retain_graph=True)
    structure_student_optimizer.step()

    feature_student_optimizer.zero_grad()
    feature_student_distillation_loss.backward(retain_graph=True)
    feature_student_optimizer.step()

    # Calculate accuracy (assuming a function calculate_accuracy(logits, labels) is defined)
    structure_teacher_acc = calculate_accuracy(structure_teacher_logits[g.ndata['train_mask']], g.ndata['label'][g.ndata['train_mask']])
    feature_teacher_acc = calculate_accuracy(feature_teacher_logits[g.ndata['train_mask']], g.ndata['label'][g.ndata['train_mask']])
    structure_student_acc = calculate_accuracy(structure_student_logits[g.ndata['train_mask']], g.ndata['label'][g.ndata['train_mask']])
    feature_student_acc = calculate_accuracy(feature_student_logits[g.ndata['train_mask']], g.ndata['label'][g.ndata['train_mask']])

    if epoch % 10 == 0:
        print(f"Epoch {epoch}")
        print(f"Structure Teacher Loss: {structure_teacher_loss.item()}, Acc: {structure_teacher_acc}")
        print(f"Feature Teacher Loss: {feature_teacher_loss.item()}, Acc: {feature_teacher_acc}")
        print(f"Structure Student Loss: {structure_student_distillation_loss.item()}, Acc: {structure_student_acc}")
        print(f"Feature Student Loss: {feature_student_distillation_loss.item()}, Acc: {feature_student_acc}")