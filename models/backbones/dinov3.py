from pathlib import Path
import sys

import torch
import torch.nn as nn
from safetensors.torch import load_file as safetensors_load
# DINOv3 backbone channels
DINOV3_ARCHS = {
    'dinov3_vits16': 384,
    'dinov3_vitb16': 768,
    'dinov3_vitl16': 1024,
}
class DINOv3(nn.Module):
    """
    DINOv3 wrapper that outputs:
        - f_map: [B, C, H/16, W/16]
        - cls_token: [B, C]
    Compatible with SALAD aggregation module.
    """

    def __init__(
        self,
        model_name='dinov3_vitl16',
        num_trainable_blocks=6,
        norm_layer=False,
        return_token=False,
        pretrained=False,
        pretrained_path=None
    ):
        super().__init__()

        assert model_name in DINOV3_ARCHS.keys()
        '''
        # load backbone
        self.model = torch.hub.load(
            './facebookresearch/dinov3',
            model_name,
            source='local',
            weights='/data/xulj/salad_tz/checkpoints/dinov3_vitl16_pretrain_sat493m.pth'
        ) 
          
        '''
           
        
        self.num_channels = DINOV3_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token
        self.pretrained_path = pretrained_path
        self.pretrained=pretrained
        package_root = Path(__file__).resolve().parents[2] / "facebookresearch" / "dinov3"
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))

        from dinov3.hub.backbones import dinov3_vitb16, dinov3_vitl16, dinov3_vits16  # type: ignore

        model_map = {
            "dinov3_vits16": dinov3_vits16,
            "dinov3_vitb16": dinov3_vitb16,
            "dinov3_vitl16": dinov3_vitl16,
        }
        if self.pretrained_path is not None:
            self.model = model_map[model_name](pretrained=False)
            try:
                state_dict = torch.load(self.pretrained_path, map_location="cpu", weights_only=True)
            except TypeError:
                state_dict = torch.load(self.pretrained_path, map_location="cpu")
            if isinstance(state_dict, dict):
                if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
                    state_dict = state_dict["state_dict"]
                elif "model" in state_dict and isinstance(state_dict["model"], dict):
                    state_dict = state_dict["model"]
            if any(key.startswith("module.") for key in state_dict):
                state_dict = {key[len("module.") :]: value for key, value in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
        else:
            self.model = model_map[model_name](pretrained=self.pretrained)
        self.num_register_tokens = getattr(self.model, "n_storage_tokens", 0)
    def forward(self, x):
        B, C, H, W = x.shape
        assert H % 16 == 0 and W % 16 == 0
        n_storage = self.num_register_tokens
        # prepare tokens
        x, hw = self.model.prepare_tokens_with_masks(x)
        # token shape now: [B, 1 + R + HW, C]
        if self.model.rope_embed is not None:
                rope_sincos = self.model.rope_embed(H=hw[0], W=hw[1])
        else:
                rope_sincos = None
        # ---- frozen blocks ----
        
        with torch.no_grad():
            for blk in self.model.blocks[:-self.num_trainable_blocks]:
                x = blk(x, rope_sincos)

        # ---- trainable blocks ----
        for blk in self.model.blocks[-self.num_trainable_blocks:]:
            x = blk(x, rope_sincos)
        
        '''
        with torch.no_grad():
            for blk in self.model.blocks:
                x = blk(x, rope_sincos)
        '''

        # ---- final normalization ----
        if self.norm_layer:
            x = self.model.norm(x)

        # ---------------------------------------------------------
        # 1. get CLS
        # ---------------------------------------------------------
        cls_token = x[:, 0]                           # [B, C]

        # ---------------------------------------------------------
        # 2. remove register tokens → keep only patch tokens
        # ---------------------------------------------------------
        # x = [CLS] + [REG1..REGr] + [patch tokens]
        patch_tokens = x[:,1 + n_storage:,:]   # [B, HW, C]

        # ---------------------------------------------------------
        # 3. reshape patch tokens -> feature map   [B, C, H/16, W/16]
        # ---------------------------------------------------------
        patch_tokens = patch_tokens.reshape(
            B, hw[0], hw[1], self.num_channels
        ).permute(0, 3, 1, 2)
        if self.return_token:
            return patch_tokens, cls_token
        return patch_tokens
