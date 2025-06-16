import torch
import torch.nn as nn
import torch.nn.functional as F

class SALAD(nn.Module):
    """
    在原 SALAD 基础上，用 VLAD-BuFF 的 burst-aware 权重
    替换掉原先的 Sinkhorn 最优传输部分。
    """
    def __init__(self,
                 num_channels=1536,
                 num_clusters=64,
                 cluster_dim=128,
                 token_dim=256,
                 dropout=0.3):
        super().__init__()
        self.num_channels = num_channels
        self.num_clusters = num_clusters  # m
        self.cluster_dim = cluster_dim    # l
        self.token_dim = token_dim        # g

        # 可选的 dropout
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        # 全局场景 token g 的 MLP
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim)
        )
        # 局部特征 f_i 的 MLP（输出维度 = cluster_dim）
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            self.dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1)
        )
        # 对每个像素点打分的 MLP（输出维度 = num_clusters）
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            self.dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )

        # 以下三个参数用于 burst-aware 权重计算
        # w_i = sum_j sigmoid(a * d_ij + b), 最后除以 w_i^p
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.p = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """
        输入:
          x: tuple(f_map, token)
            f_map: torch.Tensor, shape [B, C, H, W]
            token: torch.Tensor, shape [B, C]          （CLS token）
        返回:
          global_descriptor: torch.Tensor, shape [B, m*l + g]
        """
        f_map, token = x

        # 1) 计算局部特征 f，展平到 [B, l, N]
        #    N = H*W
        f = self.cluster_features(f_map).flatten(2)           # B, l, N

        # 2) 计算打分 scores，并做 Softmax 得到初始 assignment α  [B, m, N]
        scores = self.score(f_map).flatten(2)                 # B, m, N
        alpha = torch.softmax(scores, dim=1)                  # B, m, N

        # 3) 计算 burstiness 权重 w_i
        #    3.1 先 L2 归一化 f → [B, l, N]
        f_norm = F.normalize(f, p=2, dim=1)                   # B, l, N
        #    3.2 计算相似度矩阵 sim_{ij} = ⟨f_i, f_j⟩  → [B, N, N]
        sim = torch.matmul(f_norm.transpose(1,2), f_norm)     # B, N, N
        #    3.3 软计数: w_i = sum_j sigmoid(a * sim_{ij} + b)  → [B, N]
        w = torch.sigmoid(self.a * sim + self.b).sum(dim=2)    # B, N

        # 4) 用 w_i^p 对 α 做除法，得到 burst-discounted assignment p  [B, m, N]
        denom = w.pow(self.p).unsqueeze(1)                    # B, 1, N
        p = alpha / (denom + 1e-6)                            # B, m, N

        # 5) 同原 SALAD，拼接全局 token 和加权求和后的局部聚合
        #    5.1 归一化 global token
        t = self.token_features(token)                        # B, g
        t_norm = F.normalize(t, p=2, dim=-1)                  # B, g

        #    5.2 将 p 撑到 [B, l, m, N]，f 撑到 [B, l, m, N]
        p_exp = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)   # B, l, m, N
        f_exp = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)  # B, l, m, N

        #    5.3 对每个 cluster 做加权求和，得 [B, l, m]
        v = (f_exp * p_exp).sum(dim=-1)                       # B, l, m

        #    5.4 L2 归一化后 flatten，再与全局 token 拼接
        v = F.normalize(v, p=2, dim=1).flatten(1)             # B, l*m
        out = torch.cat([t_norm, v], dim=1)                   # B, g + l*m

        # 6) 最后做一次整体归一化
        return F.normalize(out, p=2, dim=-1)