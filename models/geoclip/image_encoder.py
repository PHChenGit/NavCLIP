import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, LayerNorm



class CrossAttention(nn.Module):
    """
    Cross-Attention Module to fuse reference image features into query image features.
    """
    def __init__(self, dim=768):
        super(CrossAttention, self).__init__()
        self.cross_attention = MultiheadAttention(embed_dim=dim, num_heads=8)
        self.layer_norm = LayerNorm(dim)

    def forward(self, query, key, value):
        """
        Args:
            query (Tensor): Query features (from query image). Shape: [seq_len, B, dim]
            key (Tensor): Key features (from reference image). Shape: [seq_len, B, dim]
            value (Tensor): Value features (from reference image). Shape: [seq_len, B, dim]

        Returns:
            Tensor: Fused features. Shape: [seq_len, B, dim]
        """
        x1 = query
        attn_output, _ = self.cross_attention(query, key, value)
        x = x1 + attn_output
        x = self.layer_norm(x)
        return x

class CrossAttention(nn.Module):
    """
    Cross-Attention Module to fuse reference image features into query image features.
    """
    def __init__(self, dim=768):
        super(CrossAttention, self).__init__()
        self.cross_attention = MultiheadAttention(embed_dim=dim, num_heads=8)
        self.layer_norm = LayerNorm(dim)

    def forward(self, query, key, value):
        """
        Args:
            query (Tensor): Query features (from query image). Shape: [seq_len, B, dim]
            key (Tensor): Key features (from reference image). Shape: [seq_len, B, dim]
            value (Tensor): Value features (from reference image). Shape: [seq_len, B, dim]

        Returns:
            Tensor: Fused features. Shape: [seq_len, B, dim]
        """
        x1 = query
        attn_output, _ = self.cross_attention(query, key, value)
        x = x1 + attn_output
        x = self.layer_norm(x)
        return x

class ImageEncoder(nn.Module):
    def __init__(self, dim = 768):
        super(ImageEncoder, self).__init__()
        self.self_attention = MultiheadAttention(embed_dim=dim, num_heads=8)
        # The original layer_norm is renamed to avoid conflicts
        self.layer_norm1 = nn.LayerNorm(dim)
        # Add the new CrossAttention module
        self.cross_attention = CrossAttention(dim=dim)

        self.mlp = nn.Sequential(nn.Linear(dim, dim),
                                 nn.ReLU(),
                                #  nn.Dropout(p=0.3),
                                 nn.Linear(dim, 512),
                                 nn.ReLU())

    def forward(self, query_feat, ref_feat):
        """
        Processes query and reference image features.

        Args:
            query_feat (Tensor): Features from the query image. Shape: [B, 768]
            ref_feat (Tensor): Features from the reference image. Shape: [B, 768]

        Returns:
            Tensor: Fused and processed features. Shape: [B, 512]
        """
        # Reshape for multi-head attention: [seq_len, B, embed_dim]
        # 1. Self-attention on query features
        q = query_feat.unsqueeze(1).permute(1, 0, 2)
        q_res = q
        q, _ = self.self_attention(q, q, q)
        q = q + q_res
        q = self.layer_norm1(q)

        # 2. Prepare key/value from reference features for cross-attention
        k = ref_feat.unsqueeze(1).permute(1, 0, 2)
        v = k
        
        # 3. Apply cross-attention
        x = self.cross_attention(query=q, key=k, value=v)

        # 4. Final MLP layers
        x = x.permute(1, 0, 2)  # Change back to [B, seq_len, embed_dim]
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = x.mean(dim=1)
        x = self.mlp(x) # [B, 512]
        return x