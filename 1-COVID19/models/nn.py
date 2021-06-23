import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        # Define your neural network here
        # TODO: How to modify this model to achieve better performance?
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(),
                                 nn.Linear(64, 1))

        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')
        # coef of regularization item
        self.lambda1 = 0.5
        self.lambda2 = 0.01

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        ''' Calculate loss '''
        # TODO: you may implement L2 regularization here
        mse_loss = self.criterion(pred, target)
        # l1_reg = self.lambda1 * sum(
            # [torch.norm(param, 1) for param in self.parameters()])
        # l2_reg = self.lambda2 * sum(
            # [torch.norm(param, 2) for param in self.parameters()])
        # return mse_loss + l1_reg + l2_reg
        return mse_loss
