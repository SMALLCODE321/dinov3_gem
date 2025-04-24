import torch
import torch.nn as nn
import torch.nn.functional as F

class SALAD(nn.Module):
    """
    SALAD 模型基于局部特征聚合，并加入了 Burst-aware 特征加权机制。
    
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

        # --- Burst-aware 专用可学习参数 ---
        # burst_b 和 burst_p 保持不变，burst_a 将由局部自适应温度代替
        self.burst_b = nn.Parameter(torch.tensor(0.0))
        self.burst_p = nn.Parameter(torch.tensor(1.0))
        # 用于融合原始 p 与 discount 修正后 p 的残差连接参数
        self.lambda_discount = nn.Parameter(torch.tensor(0.5))
        self.eps = 1e-6  # 防止除零

        # 如果 dropout 大于0，则使用 Dropout 层
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
            nn.Conv2d(self.num_channels, 512, kernel_size=1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, kernel_size=1)
        )
        # 为局部自适应温度参数增加一个 1×1 的卷积层（输入维度与聚类维度一致）
        self.temp_predictor = nn.Conv1d(in_channels=self.cluster_dim, out_channels=1, kernel_size=1)
        
        # MLP for score 矩阵 S
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, kernel_size=1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, kernel_size=1),
        )
        # Dustbin 参数 z
        self.dust_bin = nn.Parameter(torch.tensor(1.))
    
    # 以下 Sinkhorn / log-optimal transport 部分代码保持不变
    def logsumexp_alternative(self, tensor, dim, keepdim=False):
        max_val, _ = torch.max(tensor, dim=dim, keepdim=True)
        stable_tensor = tensor - max_val
        exp_tensor = torch.exp(stable_tensor)
        sum_exp = torch.sum(exp_tensor, dim=dim, keepdim=True)
        log_sum_exp = max_val + torch.log(sum_exp)
        if not keepdim:
            log_sum_exp = log_sum_exp.squeeze(dim)
        return log_sum_exp

    def log_sinkhorn_iterations(self, Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(iters):
            u = log_mu - self.logsumexp_alternative(Z + v.unsqueeze(1), dim=2)
            v = log_nu - self.logsumexp_alternative(Z + u.unsqueeze(2), dim=1)
        return Z + u.unsqueeze(2) + v.unsqueeze(1)

    def log_optimal_transport(self, scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
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
        
        Z = self.log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
        Z = Z - norm  # multiply probabilities by (M+N)
        return Z

    def forward(self, x):
        """
        参数:
            x (tuple): 包含两部分，
                - 特征张量 x，形状为 [B, C, H, W]
                - 全局 token 张量 t，形状为 [B, C]
        
        返回:
            f (torch.Tensor): 全局描述子，形状为 [B, num_clusters * cluster_dim + token_dim]
        """
        # 分离特征和 token
        x, t = x
        
        # 得到局部特征 f 和得分 p
        # f: [B, cluster_dim, N]，其中 N = H * W
        # p: [B, num_clusters, N]
        f = self.cluster_features(x).flatten(2)
        p = self.score(x).flatten(2)
        t = self.token_features(t)

        # 用 Sinkhorn 算法对 soft-assignment 权重做归一化
        p = self.log_optimal_transport(p, self.dust_bin, iters=3)
        p = torch.exp(p)
        # 去掉最后一行 (dustbin)
        p = p[:, :-1, :]

        # --- 引入局部自适应温度参数的 Burst-aware 特征加权机制 ---
        # 1. 对 f 做 L2 归一化，得到局部特征的归一化版本
        f_norm = F.normalize(f, p=2, dim=1)  # [B, cluster_dim, N]
        
        # 2. 利用局部特征 f 计算自适应温度参数：使用 1×1 卷积层（输入 f: [B, cluster_dim, N]）
        # 输出 T 的形状为 [B, 1, N]，此处用 softplus 保证温度参数为正
        T = F.softplus(self.temp_predictor(f))  # [B, 1, N]
        
        # 3. 计算局部特征间的相似性矩阵，shape: [B, N, N]
        sim = torch.bmm(f_norm.transpose(1, 2), f_norm)  # [B, N, N]
        
        # 4. 对每个局部特征 i 计算对应温度变量 T_i，并对每对 (i, j) 取平均得到 T_avg
        T_i = T.squeeze(1)          # [B, N]
        T_avg = (T_i.unsqueeze(2) + T_i.unsqueeze(1)) / 2.0  # [B, N, N]
        
        # 5. 将自适应温度参数 T_avg 应用于相似性矩阵，计算 burst 权重
        burst_weights = torch.sigmoid(T_avg * sim + self.burst_b)  # [B, N, N]
        # 对每个局部特征 i 求和，得到 discount 的分母
        burst_weight_sum = burst_weights.sum(dim=-1)  # [B, N]
        
        # 6. 计算 discount，利用 burst_p 调制，并对 discount 做 clip 限幅
        discount = burst_weight_sum ** self.burst_p   # [B, N]
        discount = torch.clamp(discount, min=1e-3, max=1e3)
        
        # 7. 引入残差连接：融合原始 soft-assignment 权重 p 与 discount 调整后的权重
        lambda_discount = torch.sigmoid(self.lambda_discount)  # 保证在0~1区间
        p_adjusted = (1 - lambda_discount) * p + lambda_discount * (p / (discount.unsqueeze(1) + self.eps))
        p = p_adjusted
        # --- End Burst-aware ---
        
        # 对 p 扩充维度与局部特征 f 对齐，进行局部特征聚合
        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)  # [B, cluster_dim, num_clusters, N]
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)  # [B, cluster_dim, num_clusters, N]
        
        # 聚合局部特征（对每个聚类中心求和）并归一化
        local_agg = (f * p).sum(dim=-1)  # [B, cluster_dim, num_clusters]
        local_agg = F.normalize(local_agg, p=2, dim=1).flatten(1)
        token_norm = F.normalize(t, p=2, dim=-1)
        global_descriptor = torch.cat([token_norm, local_agg], dim=-1)
    
        return F.normalize(global_descriptor, p=2, dim=-1)