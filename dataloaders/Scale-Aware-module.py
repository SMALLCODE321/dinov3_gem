import torch
import torch.nn as nn

class ScaleAdaptiveWrapper(nn.Module):
    def __init__(self, vit_backbone, embed_dim=768):
        super().__init__()
        self.vit = vit_backbone  # 你的 DINOv3 / ViT 模型
        
        # 1. 尺度编码器：将分辨率数值转换为向量
        self.gsd_embedder = nn.Sequential(
            nn.Linear(1, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim)
        )
        
        # 2. 如果你的增强模拟产生了不同的通道数，可以在这里处理
        # 比如卫星图是多光谱，模拟图是RGB
        self.patch_embed = vit_backbone.patch_embed 

    def forward(self, x, gsd_value):
        """
        x: [B, C, H, W] 图像张量
        gsd_value: [B, 1] 对应的分辨率（如 10.0 或 模拟的 0.1）
        """
        # A. 生成 Patch Tokens
        x = self.patch_embed(x) # [B, N, D]
        
        # B. 生成尺度嵌入 (Scale Embedding)
        # 对分辨率取对数处理，增强对跨尺度数值的鲁棒性
        gsd_feat = self.gsd_embedder(torch.log(gsd_value + 1e-6)) # [B, 1, D]
        
        # C. 将尺度信息注入每一个 Patch
        # 这样模型在做 Self-Attention 时，每个 Token 都带着“我是什么分辨率”的信息
        x = x + gsd_feat
        
        # D. 加上标准的位置编码并送入 ViT Blocks
        # 注意：这里保持 DINOv3 的 pos_embed 逻辑
        if self.vit.pos_embed is not None:
            x = x + self.vit.pos_embed[:, 1:, :] # 排除 CLS token 的位置
            
        # 插入 CLS 并运行 Transformer
        cls_token = self.vit.cls_token + gsd_feat
        x = torch.cat((cls_token, x), dim=1)
        
        for blk in self.vit.blocks:
            x = blk(x)
        
        return self.vit.norm(x)