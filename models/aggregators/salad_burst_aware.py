import torch
import torch.nn as nn

def logsumexp_alternative(tensor, dim, keepdim=False):
    """
    替代 torch.logsumexp 的实现，使用 torch.max 和 torch.log 等基础算子。
    """
    max_val, _ = torch.max(tensor, dim=dim, keepdim=True)
    stable_tensor = tensor - max_val
    exp_tensor = torch.exp(stable_tensor)
    sum_exp = torch.sum(exp_tensor, dim=dim, keepdim=True)
    log_sum_exp = max_val + torch.log(sum_exp)
    if not keepdim:
        log_sum_exp = log_sum_exp.squeeze(dim)
    return log_sum_exp

def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability """
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - logsumexp_alternative(Z + v.unsqueeze(1), dim=2)
        v = log_nu - logsumexp_alternative(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability """
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns, bs = (m * one).to(scores), (n * one).to(scores), ((n - m) * one).to(scores)

    bins = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)
    
    couplings = torch.cat([scores, bins], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = norm.expand(n)
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z

class SALAD(nn.Module):
    """
    这份 SALAD 模型代码基于局部特征聚合，现加入 Burst-aware 特征加权机制。
    
    Attributes:
        num_channels (int): 输入特征的通道数（d）。
        num_clusters (int): 聚类中心的数量（m）。
        cluster_dim (int): 每个聚类中心的特征维度（l）。
        token_dim (int): 全局场景 token 的维度（g）。
        dropout (float): dropout 概率。
    """
    def __init__(self,
                 num_channels=1536,
                 num_clusters=64,
                 cluster_dim=128,
                 token_dim=256,
                 dropout=0.3,
                ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim

        if dropout > 0:
            dropout_layer = nn.Dropout(dropout)
        else:
            dropout_layer = nn.Identity()
        
        # MLP for global scene token g
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim)
        )
        # MLP for local features f_i
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1)
        )
        # MLP for score矩阵 S
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )
        # Dustbin参数 z
        self.dust_bin = nn.Parameter(torch.tensor(1.))

        # 以下为 Burst-aware 加权机制相关的可学习参数：
        self.burst_a = nn.Parameter(torch.tensor(1.0))  # sigmoid 的斜率
        self.burst_b = nn.Parameter(torch.tensor(0.0))  # sigmoid 的偏置
        self.burst_p = nn.Parameter(torch.tensor(1.0))  # 权重的幂次

    def forward(self, x):
        """
        参数:
            x (tuple): 包含两部分，
                - 特征张量 x，形状为 [B, C, H, W]
                - 全局 token 张量 t，形状为 [B, C]
        
        返回:
            f (torch.Tensor): 全局描述子，形状为 [B, num_clusters * cluster_dim + token_dim]
        """
        x, t = x  # 分离特征和 token
        
        # 得到局部特征 f 和得分 p
        # f: [B, cluster_dim, N]，其中 N = H * W
        # p: [B, num_clusters, N]
        f = self.cluster_features(x).flatten(2)
        p = self.score(x).flatten(2)
        t = self.token_features(t)

        # Sinkhorn算法对 soft-assignment 权重进行归一化
        p = log_optimal_transport(p, self.dust_bin, 3)
        p = torch.exp(p)
        # 去掉最后一行 (dustbin)
        p = p[:, :-1, :]

        # --- Burst-aware 特征加权机制 ---
        # 计算用于 discount 的 burstiness 权重：
        # 1. 对 f 进行 L2归一化，shape: [B, cluster_dim, N]
        f_norm = nn.functional.normalize(f, p=2, dim=1)
        # 2. 计算局部特征间的相似性矩阵，shape: [B, N, N]
        sim = torch.bmm(f_norm.transpose(1, 2), f_norm)
        # 3. 用 sigmoid 及 learnable 参数对相似性进行“软计数”
        burst_weights = torch.sigmoid(self.burst_a * sim + self.burst_b)  # [B, N, N]
        burst_weight_sum = burst_weights.sum(dim=-1)  # 对每个局部特征 i 求和: [B, N]
        # 4. 计算 discount 因子
        discount = burst_weight_sum ** self.burst_p  # [B, N]
        # 融入 burst-aware 加权：调整每个局部特征的 soft-assignment 权重 p
        p = p / discount.unsqueeze(1)  # broadcast 到 [B, num_clusters, N]
        # --- End Burst-aware ---

        # 后续过程与原来 SALAD 稍有相似性：对局部特征聚合与全局 token 拼接
        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)  # [B, cluster_dim, num_clusters, N]
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)  # [B, cluster_dim, num_clusters, N]

        # 聚合局部特征 (f * p) 并 summing over局部区域
        local_agg = (f * p).sum(dim=-1)  # [B, cluster_dim, num_clusters]
        local_agg = nn.functional.normalize(local_agg, p=2, dim=1).flatten(1)
        token_norm = nn.functional.normalize(t, p=2, dim=-1)
        global_descriptor = torch.cat([token_norm, local_agg], dim=-1)
        
        return nn.functional.normalize(global_descriptor, p=2, dim=-1)