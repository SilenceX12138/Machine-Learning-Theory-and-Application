from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention, ReLU, SiLU
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm


class ConvClassifier(nn.Module):
    def __init__(self, d_model=80, n_spks=600, dropout=0.1):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)  # require last dim of data is 40
        self.encoder_layer = ConformerEncoderLayer(
            d_model=d_model,  # d_model: size of encoder
            dim_feedforward=1024,
            nhead=2)

        # Project the the dimension of features from d_model into speaker nums.
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels):
        """
    args:
      mels: (batch size, length, 40)
    return:
      out: (batch size, n_spks)
    """
        # out: (batch size, length, d_model)
        out = self.prenet(mels)
        # out: (length, batch size, d_model)
        out = out.permute(1, 0, 2)  # reset sequence of dimensions
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        out = self.encoder_layer(out)
        # out = self.encoder(out)
        # out: (batch size, length, d_model)
        out = out.transpose(0, 1)  # exchange dim 0 and dim 1
        # stats: (batch size, d_model)
        stats = out.mean(dim=1)  # mean pooling

        # out: (batch, n_spks)
        out = self.pred_layer(stats)
        return out


class ConformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(d_model,
                                               dim_feedforward), ReLU(),
                                     Dropout(dropout),
                                     nn.Linear(dim_feedforward, d_model),
                                     ReLU(), Dropout(dropout))
        self.dropout1 = Dropout(dropout)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout2 = Dropout(dropout)
        self.conv_block = ConformerModule(d_model)
        self.linear2 = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            ReLU(),
            Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            ReLU(),
            Dropout(dropout),
        )
        self.dropout3 = Dropout(dropout)

        self.norm = LayerNorm(d_model)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
            self,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.linear1(src)
        src = src + 0.5 * self.dropout1(src2)
        src2 = self.self_attn(src,
                              src,
                              src,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout2(src2)
        src2 = self.conv_block(src)
        src = src + src2  # dropout is alrealy executed within conv_block
        src2 = self.linear2(src)
        src = src + 0.5 * self.dropout3(src2)
        src = self.norm(src)
        return src


class ConformerModule(nn.Module):
    def __init__(self,
                 d_model,
                 kernel_size=3,
                 expansion_factor=2,
                 dropout=0.1):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.point_conv1 = nn.Conv1d(d_model,
                                     d_model * expansion_factor,
                                     kernel_size=1)
        self.act1 = _get_activation_fn('glu')
        self.dep_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            # same convolution
            padding=(kernel_size - 1) // 2)
        self.norm2 = nn.BatchNorm1d(
            d_model, affine=False)  # without learnable parameters
        self.act2 = _get_activation_fn('swish')
        self.point_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = Dropout(dropout)

    def forward(self, src):
        src2 = self.norm1(src) # layernorm takes last dimension as default channel dimension

        src2 = src2.permute(0, 2, 1)
        src2 = self.point_conv1(src2)
        src2 = self.act1(src2, 1)
        src2 = src2.permute(0, 2, 1)

        src2 = src2.permute(0, 2, 1)
        src2 = self.dep_conv(src2)
        src2 = self.norm2(src2)  # batchnorm takes dim[1] as channel dimension
        src2 = src2.permute(0, 2, 1)

        src2 = self.act2(src2)

        src2 = src2.permute(0, 2, 1)
        src2 = self.point_conv2(src2)
        src2 = src2.permute(0, 2, 1)

        src2 = self.dropout(src2)
        src = src + src2
        return src


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == 'swish':
        return F.silu
    elif activation == 'glu':
        return F.glu

    raise RuntimeError(
        "activation should be relu/gelu, not {}".format(activation))
