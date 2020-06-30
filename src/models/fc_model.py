import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class BaseModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=3, hidden_sizes=200):
        super(BaseModel, self).__init__()
        if np.isscalar(hidden_sizes):
            hidden_sizes = [hidden_sizes] * hidden_layers
        sizes = [input_size, *hidden_sizes, output_size]
        self.linears = nn.ModuleList([nn.Linear(in_size, out_size)
                                      for in_size, out_size in zip(sizes, sizes[1:])])

    def forward(self, x):

        x = torch.flatten(x,1)
        for layer in self.linears:
            x = layer(x)
            if layer != self.linears[-1]:
                x = F.relu(x)

        return x

