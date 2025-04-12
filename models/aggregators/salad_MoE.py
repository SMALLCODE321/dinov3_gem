import torch
import torch.nn as nn

def logsumexp_alternative(tensor, dim, keepdim=False):
    """
    替代 torch.logsumexp 的实现，使用 torch.max 和 torch.log 等基本算子
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
    """ 使用对数空间进行 Sinkhorn 归一化操作，提升数值稳定性 """
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - logsumexp_alternative(Z + v.unsqueeze(1), dim=2)
        v = log_nu - logsumexp_alternative(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ 使用对数空间计算可微分的最优传输 """
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms = (m * one).to(scores)
    ns = (n * one).to(scores)
    # 这里构造额外的列作为 dustbin
    bins = alpha.expand(b, 1, n)
    # 在这里保证 alpha 的形状正确
    couplings = torch.cat([scores, bins], dim=1)

    norm = - (ms + ns).log()
    # 构建目标边缘 log_mu 和 log_nu
    log_mu = torch.cat([norm.expand(m), (ns.log() + norm)[None]])
    log_nu = norm.expand(n)
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # 恢复原来的数值幅度
    return Z

class MoEScore(nn.Module):
    """
    MoE 得分模块，包含多个专家用于处理不同倾角下的特征。
    输入：
      x: 特征图，形状为 [B, C, H, W]
      token: 全局 token 特征，形状为 [B, C_global]，可用于门控网络获得专家权重
    输出：
      得分张量，形状为 [B, num_clusters, H, W]
    """
    def __init__(self, num_channels, num_clusters, num_experts=2, dropout=0.3):
        super().__init__()
        self.num_experts = num_experts
        # 定义多个专家
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_channels, 512, kernel_size=1),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.ReLU(),
                nn.Conv2d(512, num_clusters, kernel_size=1)
            ) for _ in range(num_experts)
        ])
        # 门控网络：利用全局 token 输出每个专家的权重
        self.gate = nn.Sequential(
            nn.Linear(num_channels, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts)
        )

    def forward(self, x, token):
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))  # 每个输出形状 [B, num_clusters, H, W]
        # 利用 token（可能携带倾角信息）获得每个专家的权重
        gate_weights = self.gate(token)  # [B, num_experts]
        gate_weights = torch.softmax(gate_weights, dim=1)
        # 每个专家的输出加权组合
        combined = 0
        for i in range(self.num_experts):
            # 扩展权重至 [B, 1, 1, 1] 与专家输出相乘
            combined = combined + gate_weights[:, i].view(-1, 1, 1, 1) * expert_outputs[i]
        return combined

class SALAD(nn.Module):
    """
    SALAD 模型（Sinkhorn Algorithm for Locally Aggregated Descriptors）结合了 MoE 得分模块。
    
    Attributes:
        num_channels (int): 输入特征的通道数。
        num_clusters (int): 聚类簇数目。
        cluster_dim (int): 聚类簇特征通道数，用于计算局部特征。
        token_dim (int): 全局 scene token 的维度。
        dropout (float): dropout 率。
    """
    def __init__(self,
                 num_channels=1536,
                 num_clusters=64,
                 cluster_dim=128,
                 token_dim=256,
                 dropout=0.3,
                 num_experts=2  # MoE 中的专家数量
                 ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim

        # 定义 dropout 层，如果 dropout=0，则 Identity
        dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 全局 scene token 的 MLP
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim)
        )
        # 局部特征提取：局部聚类特征
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, kernel_size=1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, kernel_size=1)
        )
        # 使用 MoE 模块来生成得分矩阵 S
        self.score = MoEScore(num_channels, num_clusters, num_experts=num_experts, dropout=dropout)

        # Dustbin 参数，用于最优传输中处理未匹配的情况
        self.dust_bin = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        """
        输入：
          x (tuple): 包含两个元素
               - 特征图：形状 [B, C, H, W]
               - 全局 token：形状 [B, C]，这里包含了全局场景信息及可能的倾角信息
        过程：
          1. 对输入特征进行局部和全局分支提取。
          2. 使用 MoE 得分模块得到得分矩阵，再利用 Sinkhorn 最优传输。
          3. 最后将全局特征和加权聚合的局部特征拼接后归一化。
        输出：
          全局描述子，形状为 [B, m*l + g]
        """
        x_feat, token = x  # 提取局部特征和全局 token
        # 计算局部聚类特征，并将空间维度展平 [B, cluster_dim, H*W]
        f = self.cluster_features(x_feat).flatten(2)
        # 利用 MoE 模块计算 score，注意此处同时输入 x_feat 和全局 token 以获得专家加权
        p = self.score(x_feat, token).flatten(2)  # [B, num_clusters, H*W]
        # 计算全局 token 特征（全局描述子）
        t = self.token_features(token)
        
        # Sinkhorn 最优传输
        p = log_optimal_transport(p, self.dust_bin, iters=3)
        p = torch.exp(p)
        # 截断掉 dustbin 部分（最后一行）
        p = p[:, :-1, :]

        # 将 p 和局部聚类特征 f 做组合
        # 先将 p 扩展到与专家维度对齐
        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)
        # 将全局 token 和局部特征加权求和后拼接，并归一化
        f = torch.cat([
            nn.functional.normalize(t, p=2, dim=-1),
            nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1)
        ], dim=-1)
        return nn.functional.normalize(f, p=2, dim=-1)