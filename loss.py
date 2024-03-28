import torch
import torch.nn as nn
import torch.nn.functional as F

def distillation_loss(teacher_logits, student_logits, temperature):
    """
    计算知识蒸馏损失。
    teacher_logits: 教师模型的输出
    student_logits: 学生模型的输出
    temperature: 温度参数，用于软化概率分布
    """
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    return loss
