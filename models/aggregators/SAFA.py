import torch
import torch.nn as nn
import torch.nn.functional as F

class SAFA(nn.Module):
    """
    Self-Attention Feature Aggregation
    输入: patch map (B, C, H, W)
    输出: 聚合向量 (B, C)
    """
    def __init__(self, in_channels, attn_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.attn_dim = attn_dim or in_channels

        # QKV 线性变换
        self.q_proj = nn.Linear(in_channels, self.attn_dim)
        self.k_proj = nn.Linear(in_channels, self.attn_dim)
        self.v_proj = nn.Linear(in_channels, self.attn_dim)

        # 可选输出线性映射回 C
        self.out_proj = nn.Linear(self.attn_dim, in_channels)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.view(B, C, N).permute(0, 2, 1)  # (B, N, C)

        # QKV
        Q = self.q_proj(x_flat)  # (B, N, D)
        K = self.k_proj(x_flat)  # (B, N, D)
        V = self.v_proj(x_flat)  # (B, N, D)

        # Attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.attn_dim ** 0.5)  # (B, N, N)
        attn = F.softmax(attn, dim=-1)

        # 聚合
        agg = torch.matmul(attn, V)  # (B, N, D)
        agg = agg.mean(dim=1)         # (B, D)

        # 映射回原始通道数
        agg = self.out_proj(agg)      # (B, C)
        return agg
