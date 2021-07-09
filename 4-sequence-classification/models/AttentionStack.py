import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionStack(nn.Module):
    def __init__(self, d_model=80, n_spks=600, dropout=0.1):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model) # require last dim of data is 40
        # TODO:
        #   Change Transformer to Conformer.
        #   https://arxiv.org/abs/2005.08100

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,  # d_model: size of encoder
            dim_feedforward=256,
            nhead=2)
        # use this function to stack attention layers
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

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
        out = self.encoder(out)
        # out: (batch size, length, d_model)
        out = out.transpose(0, 1)  # exchange dim 0 and dim 1
        # stats: (batch size, d_model)
        stats = out.mean(dim=1)  # mean pooling

        # out: (batch, n_spks)
        out = self.pred_layer(stats)
        return out
