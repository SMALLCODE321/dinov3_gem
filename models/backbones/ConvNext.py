import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt as a drop-in backbone for SALAD:
      - f_map: [B, C, H', W']  （最后stage的特征图）
      - token: [B, C]         （全局avg‐pooled向量）
    Args:
        model_name (str): 如 "convnext_tiny", "convnext_small", "convnext_base", "convnext_large"
        pretrained (bool)
        out_index (int): timm.features_only 输出哪一层，ConvNeXt有4个stage，对应索引0~3，通常取3
    """
    def __init__(self,
                 model_name: str = "convnext_base",
                 pretrained: bool = True,
                 out_index: int = 3):
        super().__init__()
        # features_only=True 返回一个 list of feature maps，对应 out_indices
        self.body = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(out_index,)
        )
        # 拿到最后一层的通道数，方便后续构造SALAD
        # timm >=0.6.7: self.body.feature_info.channels() 是 list
        self.num_channels = self.body.feature_info.channels()[out_index]

    def forward(self, x: torch.Tensor):
        """
        x: [B,3,H,W]
        returns:
          f_map: [B, C, H', W']
          token: [B, C]
        """
        feats = self.body(x)      # list size=1
        f_map = feats[0]          # [B, C, H', W']
        # 全局token 用 avgpool
        token = F.adaptive_avg_pool2d(f_map, (1,1)).flatten(1)  # [B, C]
        return f_map, token