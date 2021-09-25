import torch
from torch.autograd import Function
from config import train_config


class GRL(Function):
    @staticmethod
    def forward(self, x):
        return x

    @staticmethod
    def backward(self, grad_output):
        grad_output = train_config.lamb * grad_output.neg()
        return grad_output