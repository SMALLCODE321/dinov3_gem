import torch
import torch.onnx
from torch import nn

DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

class DINOv2(nn.Module):
    def __init__(
            self,
            model_name='dinov2_vitb14',
            num_trainable_blocks=2,
            norm_layer=False,
            return_token=False
        ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        self.model = torch.hub.load('facebookresearch/dinov2', model_name, source='local', pretrained=False)
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.model.prepare_tokens_with_masks(x)
        
        # First blocks are frozen
        with torch.no_grad():
            for blk in self.model.blocks[:-self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach()

        # Last blocks are trained
        for blk in self.model.blocks[-self.num_trainable_blocks:]:
            x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)
        
        t = x[:, 0]
        f = x[:, 1:]

        # Reshape to (B, C, H, W)
        f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)

        if self.return_token:
            return f, t
        return f


# 加载权重
model = DINOv2(return_token=True)
model.load_state_dict(torch.load("dinov2_weight.pth"))

# 确保将模型移动到 GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # 检查是否有 GPU
model = model.to(device)  # 移动模型到 GPU 或 CPU
model.eval()  # 切换为评估模式

# 定义输入数据格式
batch_size = 1  # 可调整
channels = 3
height = 224  # 必须是 14 的倍数
width = 224  # 必须是 14 的倍数

# 构造输入张量，并确保其在同一设备上
x = torch.randn(batch_size, channels, height, width).to(device)  # 将输入张量移动到与模型相同的设备

# 转换为 ONNX 格式
onnx_file_path = "dinov2_model.onnx"
torch.onnx.export(
    model,
    x,  # 输入张量
    onnx_file_path,  # 输出 ONNX 文件路径
    export_params=True,  # 导出权重
    opset_version=13,  # ONNX opset 版本
    input_names=["input"],  # 输入名称
    output_names=["feature_map", "token"] if model.return_token else ["feature_map"],  # 输出名称
    dynamic_axes={  # 动态轴支持
        "input": {0: "batch_size"},  # 批量大小可变
        "feature_map": {0: "batch_size"},  # 输出批量大小可变
        "token": {0: "batch_size"} if model.return_token else {},  # Token 动态批量大小
    }
)

print(f"ONNX model has been saved to {onnx_file_path}")

