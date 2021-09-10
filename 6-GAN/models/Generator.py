import torch.nn as nn
from utils.model import weights_init


class Generator(nn.Module):
    """
    Input shape: (N, in_dim)
    Output shape: (N, 3, 64, 64)
    """
    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()

        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim,
                                   out_dim,
                                   5,
                                   2,
                                   padding=2,
                                   output_padding=1,
                                   bias=False), nn.BatchNorm2d(out_dim),
                nn.ReLU())

        self.l1 = nn.Sequential(nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
                                nn.BatchNorm1d(dim * 8 * 4 * 4), nn.ReLU())
        # self.l2_5 = nn.Sequential(
        #     dconv_bn_relu(dim * 8, dim * 4), dconv_bn_relu(dim * 4, dim * 2),
        #     dconv_bn_relu(dim * 2, dim),
        #     nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
        #     nn.Tanh())
        self.t1 = dconv_bn_relu(dim * 8, dim * 4)
        self.t2 = dconv_bn_relu(dim * 4, dim * 2)
        self.t3 = dconv_bn_relu(dim * 2, dim)
        self.t4 = nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1)
        self.t5 = nn.Tanh()
        self.apply(weights_init)

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        # y = self.l2_5(y)
        y = self.t1(y) # [64, 512, 4, 4] to [64, 256, 8, 8]
        y = self.t2(y) # [64, 256, 8, 8] to [64, 128, 16, 16]
        y = self.t3(y) # [64, 128, 16, 16] to [64, 64, 32, 32]
        y = self.t4(y) # [64, 64, 32, 32] to [64, 3, 64, 64]
        y = self.t5(y)
        return y
