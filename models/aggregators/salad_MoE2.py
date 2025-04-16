import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import multiprocessing as mp

def identity(x):
    return x

class Mlp(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_sizes,
            hidden_activation=F.relu,
            output_activation=identity,
            layer_norm=True,
            out_layer_norm=False,
            use_residual=False,
    ):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.out_layer_norm = out_layer_norm
        self.use_residual = use_residual

        self.fcs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = self.m_init(nn.Linear(in_size, next_size))
            in_size = next_size
            self.fcs.append(fc)

            if self.layer_norm:
                ln = nn.LayerNorm(next_size)
                self.layer_norms.append(ln)

        self.last_fc = self.m_init(nn.Linear(in_size, output_size))
        if self.out_layer_norm:
            self.last_ln = nn.LayerNorm(output_size)

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x1 = fc(x)
            if self.layer_norm:
                x1 = self.layer_norms[i](x1)
            if self.use_residual and (x.shape[-1] == x1.shape[-1]):
                x = x + self.hidden_activation(x1)
            else:
                x = self.hidden_activation(x1)

        y = self.last_fc(x)
        if self.out_layer_norm:
            y = self.last_ln(y)

        if self.use_residual and (x.shape[-1] == y.shape[-1]):
            y = x + self.output_activation(y)
        else:
            y = self.output_activation(y)
        return y

    def m_init(self, module, gain=0.01, activate=False):
        if activate:
            gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(module.weight.data, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)
        return module

class SoftMoE(nn.Module):
    def __init__(self, d_model, num_experts, num_slots):
        super().__init__()
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.experts = nn.ModuleList([
            Mlp(d_model, d_model, [d_model],
                hidden_activation=F.relu, output_activation=identity,
                layer_norm=True, out_layer_norm=False, use_residual=False)
            for _ in range(num_experts)
        ])
        self.phi = nn.Parameter(torch.randn(d_model, num_experts, num_slots))

    def forward(self, x, mask=None):
        # x.shape: [B, N, D]， mask.shape: [B, N]; e: experts, s: slots
        weights = torch.einsum("b n d, d e s -> b n e s", x, self.phi)
        if mask is not None:
            mask = einops.rearrange(mask, "b n -> b n 1 1")
            weights = weights.masked_fill(~mask, -torch.finfo(weights.dtype).max)

        # dispatch tokens to experts
        dispatch_weights = F.softmax(weights, dim=1)
        experts_inputs = torch.einsum("b n e s, b n d -> b e s d", dispatch_weights, x)
        # 每个 expert 分别处理对应的 tokens
        expert_outputs = torch.stack([self.experts[i](experts_inputs[:, i]) for i in range(self.num_experts)])
        expert_outputs = einops.rearrange(expert_outputs, "e b s d -> b (e s) d")
        # combine expert outputs
        combine_weights = einops.rearrange(weights, "b n e s -> b n (e s)")
        combine_weights = F.softmax(combine_weights, dim=-1)
        out = torch.einsum("b n z, b z d -> b n d", combine_weights, expert_outputs)
        return out

class SoftMoEScore(nn.Module):
    """
    使用 SoftMoE 生成得分，与原 MoEScore 功能类似。
    输入：
      x: 特征图，形状为 [B, C, H, W]
      token: 全局 token 特征，形状为 [B, token_dim]（可选，若提供会经过投影后与 x 进行融合）
    输出：
      得分张量，形状为 [B, num_clusters, H, W]
    """
    def __init__(self, num_channels, num_clusters, num_experts=2, dropout=0.3, token_dim=None):
        super().__init__()
        # 利用卷积将输入特征投影到 num_clusters 维度
        self.proj = nn.Sequential(
            nn.Conv2d(num_channels, num_clusters, kernel_size=1),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        # 采用 SoftMoE（d_model 和 num_slots 均为 num_clusters）
        self.softmoe = SoftMoE(d_model=num_clusters, num_experts=num_experts, num_slots=num_clusters)
        # 如果提供 token 信息，则投影 token 到 num_clusters 维度后融合
        if token_dim is not None:
            self.token_proj = nn.Linear(token_dim, num_clusters)
        else:
            self.token_proj = None

    def forward(self, x, token):
        """
        x: [B, num_channels, H, W]
        token: [B, token_dim]
        """
        x_proj = self.proj(x)  # [B, num_clusters, H, W]
        if self.token_proj is not None:
            token_feat = self.token_proj(token)  # [B, num_clusters]
            # 融合 token 信息，广播到每个空间位置
            x_proj = x_proj + token_feat.unsqueeze(-1).unsqueeze(-1)
        
        B, C, H, W = x_proj.shape
        x_flat = x_proj.view(B, C, -1).transpose(1, 2)  # [B, H*W, num_clusters]
        out = self.softmoe(x_flat)  # [B, H*W, num_clusters]
        out = out.transpose(1, 2).view(B, C, H, W)
        return out

############################################
# SALAD 模型（移除最优传输直接使用 MoE 得分）
############################################

class SALAD(nn.Module):
    """
    SALAD 模型（Sinkhorn Algorithm for Locally Aggregated Descriptors）
    但这里将最优传输部分移除，直接用 SoftMoE 得分进行聚合。

    Attributes:
        num_channels (int): 输入特征图通道数。
        num_clusters (int): 聚类簇数目。
        cluster_dim (int): 局部聚类特征通道数。
        token_dim (int): 全局 token 维度。
        dropout (float): dropout 率。
    """
    def __init__(self,
                 num_channels=1536,
                 num_clusters=64,
                 cluster_dim=128,
                 token_dim=256,
                 dropout=0.3,
                 num_experts=2):
        super().__init__()
        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim

        # 定义 dropout 层
        dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 全局 scene token 的 MLP
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim)
        )
        # 局部聚类特征提取
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, kernel_size=1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, kernel_size=1)
        )
        # 使用 SoftMoE 模块生成得分矩阵（不使用 OT）
        self.score = SoftMoEScore(num_channels, num_clusters, num_experts=num_experts, dropout=dropout, token_dim=self.token_dim)

    def forward(self, x):
        """
        输入：
          x (tuple): 包含两个元素：
               - 特征图：形状 [B, C, H, W]
               - 全局 token：形状 [B, C]
        过程：
          1. 对输入特征进行局部和全局分支提取。
          2. 使用 SoftMoE 得分模块直接获得得分矩阵。
          3. 对得分矩阵在聚类维度上归一化后，与局部聚类特征融合。
          4. 最后将全局 token 和加权局部特征拼接后归一化。
        输出：
          全局描述子，形状为 [B, cluster_dim*num_clusters + token_dim]
        """
        x_feat, token = x  # 提取局部特征和全局 token
        # 计算局部聚类特征，并将空间维度展平：[B, cluster_dim, H*W]
        f = self.cluster_features(x_feat).flatten(2)
        # 先通过 token_features 将原始 token 投影到 token_dim（例如 256）维度
        token_processed = self.token_features(token)  # [B, token_dim]

        # 使用 SoftMoE 得分模块直接计算得分矩阵，输出形状 [B, num_clusters, H, W]
        p = self.score(x_feat, token_processed).flatten(2)  # [B, num_clusters, H*W]
        # 此处用 softmax 对各聚类簇的得分在聚类维度上归一化
        p = F.softmax(p, dim=1)

        t = token_processed
        # 将 p 扩展至与局部特征的通道对齐
        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)  # [B, cluster_dim, num_clusters, H*W]
        f_exp = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)  # [B, cluster_dim, num_clusters, H*W]
        # 利用得分矩阵对局部特征进行加权求和
        local = (f_exp * p).sum(dim=-1)  # [B, cluster_dim, num_clusters]
        local = nn.functional.normalize(local.flatten(1), p=2, dim=1)
        t = nn.functional.normalize(t, p=2, dim=-1)
        out = torch.cat([t, local], dim=-1)
        return nn.functional.normalize(out, p=2, dim=-1)