import torch
import torch.nn as nn
from salad import SALAD

# 包装模型类以适应 ONNX 导出
class ONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t):
        # 将两个输入合并为元组，传递给原始模型
        return self.model((x, t))

# 加载模型权重
model = SALAD()
model.load_state_dict(torch.load("salad_weight.pth"))
model.eval()

# 包装原始模型
onnx_model = ONNXWrapper(model)

# 定义输入数据格式
batch_size = 1  # 可以根据需求调整
num_channels = 1536
height = 16  # 示例特征图高度
width = 16   # 示例特征图宽度
token_dim = num_channels

# 构造输入张量
x = torch.randn(batch_size, num_channels, height, width)  # 特征张量
t = torch.randn(batch_size, token_dim)  # token 张量

# 转换为 ONNX 格式
onnx_file_path = "salad_model2.onnx"
torch.onnx.export(
    onnx_model,                         # 包装后的模型
    (x, t),                             # 直接传递 x 和 t
    onnx_file_path,                     # 输出 ONNX 文件路径
    export_params=True,                 # 导出训练参数
    opset_version=11,                   # ONNX opset 版本
    input_names=["features", "tokens"], # 输入名称
    output_names=["output"],            # 输出名称
    dynamic_axes={                      # 动态轴支持
        "features": {0: "batch_size"},  # 批量大小可变
        "tokens": {0: "batch_size"},    # 批量大小可变
        "output": {0: "batch_size"}     # 输出批量大小可变
    }
)

print(f"ONNX model has been saved to {onnx_file_path}")

