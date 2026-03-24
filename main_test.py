import torch
import torch.nn as nn
from torchvision import transforms
from vpr_model import VPRModel, VPREvaluator
import os
from collections import OrderedDict

def zero_shot_test():
    # 1. 环境准备
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 定义模型路径
    ckpt_path = '/data/xulj/dinov3-salad/train_result/model/DenseUAV-5e-6-50epoch.pth'

    # 3. 实例化模型（必须先定义结构，再填入权重）
    model = VPRModel(
        backbone_arch='dinov3_vitb16',
        backbone_config={
            'num_trainable_blocks': 0, # 测试时固定 backbone
            'return_token': True,
            'norm_layer': True,
            'pretrained': False, # 关闭内部自动加载，改为下面手动加载
        },
        agg_arch='GEM',
        agg_config={
            'p': 3.0, 
            'eps': 1e-6
        },
        # 以下参数在评估模式下不生效，但实例化必须提供
        lr = 6e-5,
        optimizer='adamw',
        weight_decay=9.5e-9,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {'start_factor': 1, 'end_factor': 0.2, 'total_iters': 1000},
        loss_name='MultiSimilarityLoss'
    )

    # --- 核心修改：手动加载权重并处理 DDP 前缀 ---
    if os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path}...")
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        # 如果你之前保存的是整个 model 对象而非 state_dict
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()

        # 处理 DDP 保存时产生的 'module.' 前缀
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k 
            new_state_dict[name] = v
            
        # 加载到模型中
        # strict=False 可以防止因为某些优化器参数不在 state_dict 里而报错
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"Weights loaded. Missing keys: {msg.missing_keys}")
    else:
        print("Warning: No checkpoint found. Running with initial weights.")

    model.to(device)
    model.eval()

    # 4. 执行评估
    print(">>> 启动 Zero-shot 评估 (Drone -> Satellite)...")
    
    with torch.no_grad():
        evaluator = VPREvaluator(
            model=model,
            gallery_path="/data/xulj/dinov3-salad/SUES-200-512x512/satellite-view",
            query_path="/data/xulj/dinov3-salad/SUES-200-512x512/sues200_drone_300m",
            batch_size=64, 
        )
        stats = evaluator.evaluate()

    print("\n" + "="*30)
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")
    print("="*30)

if __name__ == '__main__':
    zero_shot_test()