import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub.*')

class ImageEncoder(nn.Module):
    def __init__(self, dim = 768):
        super(ImageEncoder, self).__init__()
        self.self_attention = MultiheadAttention(embed_dim=dim, num_heads=8)

        self.mlp = nn.Sequential(nn.Linear(dim, dim),
                                 nn.ReLU(),
                                #  nn.Dropout(p=0.3),
                                 nn.Linear(dim, 512),
                                 nn.ReLU())

    def forward(self, x):
        x = x.unsqueeze(1) # [B, 1, 768]
        x = x.permute(1, 0, 2) # seq_len, B, embed_dim = [1, B, 768]
        x, _ = self.self_attention(x, x, x)
        x = x.permute(1, 0, 2)  # change back to batch, seq_len, embed_dim
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = x.mean(dim=1)
        x = self.mlp(x) # [B, 512]
        return x
