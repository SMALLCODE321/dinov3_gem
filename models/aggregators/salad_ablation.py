import torch
import torch.nn as nn
import torch.nn.functional as F

class SALAD(nn.Module):
    """
    SALAD 去掉全局 token，只对 feature map 做局部聚合，返回维度 (B, cluster_dim * num_clusters)
    """
    def __init__(self,
                 num_channels,     # ResNet 输出的通道数，比如 2048 或 512
                 num_clusters=64,  # m
                 cluster_dim=128,  # l
                 dropout=0.3):
        super().__init__()
        self.num_channels  = num_channels
        self.num_clusters  = num_clusters
        self.cluster_dim   = cluster_dim

        # 可选的 dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 局部特征 f_i 的 MLP，输出维度 = l
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            self.dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1)
        )
        # 对每个像素打分的 MLP，输出维度 = m
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            self.dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )

        # burst-aware 参数
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.p = nn.Parameter(torch.tensor(1.0))

    def forward(self, f_map):
        """
        输入:
          f_map: torch.Tensor, shape [B, C, H, W]
        返回:
          v_flat_norm: torch.Tensor, shape [B, l * m]
        """
        B, C, H, W = f_map.shape
        N = H * W

        # 1) 计算局部特征 f: [B, l, N]
        f = self.cluster_features(f_map).flatten(2)  # B, l, N

        # 2) 计算 assignment α: [B, m, N]
        scores = self.score(f_map).flatten(2)        # B, m, N
        alpha  = torch.softmax(scores, dim=1)        # B, m, N

        # 3) 计算 burstiness 权重 w_i: [B, N]
        f_norm = F.normalize(f, p=2, dim=1)          # B, l, N
        sim    = torch.matmul(f_norm.transpose(1,2), f_norm)  # B, N, N
        w      = torch.sigmoid(self.a * sim + self.b).sum(dim=2)  # B, N

        # 4) 得到 burst‐discounted assignment p: [B, m, N]
        denom = w.pow(self.p).unsqueeze(1)           # B,1,N
        p     = alpha / (denom + 1e-6)               # B, m, N

        # 5) 对每个 cluster 做加权求和
        #    扩展到 [B, l, m, N]
        p_exp = p.unsqueeze(1).expand(-1, self.cluster_dim, -1, -1)  # B,l,m,N
        f_exp = f.unsqueeze(2).expand(-1, -1, self.num_clusters, -1) # B,l,m,N

        v = (f_exp * p_exp).sum(dim=-1)   # B, l, m

        # 6) L2 归一化并 flatten
        v_flat     = v.flatten(1)        # B, l*m
        v_flat_norm = F.normalize(v_flat, p=2, dim=1)

        return v_flat_norm