import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from openTSNE import TSNE
from torchvision import transforms as T
from PIL import Image
from vpr_model import VPRModel

# ---------------------------
# 参数配置与预处理设定
# ---------------------------
IMAGENET_MEAN_STD = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}
image_size = (322, 322)
transform_drone = T.Compose([
    T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN_STD['mean'], std=IMAGENET_MEAN_STD['std']),
])

# ---------------------------
# 模型初始化及加载预训练权重
# ---------------------------
model = VPRModel(
    #---- Encoder
    backbone_arch='dinov2_vitb14',
    backbone_config={
        'num_trainable_blocks': 4,
        'return_token': True,
        'norm_layer': True,
    },
    agg_arch='SALAD',
    agg_config={
        'num_channels': 768,
        'num_clusters': 64,
        'cluster_dim': 128,
        'token_dim': 256,
    },
    lr=6e-5,
    optimizer='adamw',
    weight_decay=9.5e-9,
    momentum=0.9,
    lr_sched='linear',
    lr_sched_args={
        'start_factor': 1,
        'end_factor': 0.2,
        'total_iters': 1200,
    },
    loss_name='MultiSimilarityLoss',
    miner_name='MultiSimilarityMiner',
    miner_margin=0.1,
    faiss_gpu=False
)

pretrained_state_dict = torch.load('/data/qiaoq/Project/salad_tz/checkpoints/dino_salad.ckpt')
model.load_state_dict(pretrained_state_dict)
model = model.to('cuda')
model.eval()

# ---------------------------
# 定义函数：用于从图像中提取 patch
# ---------------------------
def extract_patches(image, patch_size, stride=None):
    """
    从 PIL.Image 格式的 image 中切出 patch，
    默认不重叠（stride=patch_size），也可指定 stride 以实现局部重叠。
    """
    if stride is None:
        stride = patch_size
    width, height = image.size
    patches = []
    patch_coords = []  # 记录patch左上角坐标
    stride_w = stride[0] if isinstance(stride, tuple) else stride
    stride_h = stride[1] if isinstance(stride, tuple) else stride
    for top in range(0, height - patch_size[1] + 1, stride_h):
        for left in range(0, width - patch_size[0] + 1, stride_w):
            patch = image.crop((left, top, left + patch_size[0], top + patch_size[1]))
            patches.append(patch)
            patch_coords.append((left, top))
    return patches, patch_coords

# ---------------------------
# 固定输入：Ground Truth 与 Satellite Patch 特征
# ---------------------------

# Ground Truth 图像
ground_truth_path = '/data/qiaoq/Project/salad_tz/demo/ground_truth.jpg'
ground_truth_img = Image.open(ground_truth_path).convert('RGB')
ground_truth_tensor = transform_drone(ground_truth_img).unsqueeze(0).to('cuda')
with torch.no_grad():
    ground_truth_feature = model(ground_truth_tensor)
ground_truth_feature = ground_truth_feature.cpu().numpy().flatten()

# 卫星图像 patch 特征
satellite_image_path = '/data/qiaoq/Project/salad_tz/demo/satellite_image.jpg'
satellite_img = Image.open(satellite_image_path).convert('RGB')
patch_size = (800, 800)
patch_stride = (400, 400)
patches, patch_coords = extract_patches(satellite_img, patch_size, stride=patch_stride)

patch_features = []
for patch in patches:
    patch_tensor = transform_drone(patch).unsqueeze(0).to('cuda')
    with torch.no_grad():
        feature = model(patch_tensor)
    patch_features.append(feature.cpu().numpy().flatten())
patch_features = np.array(patch_features)

# 构建固定基础特征集：这里第一行为 Ground Truth，其余行为各个卫星 patch
fixed_features = np.vstack([
    ground_truth_feature[np.newaxis, :],
    patch_features
])

# 使用 openTSNE 计算固定基础点的 TSNE 嵌入
tsne_fixed = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    initialization="pca"
)
fixed_embedding = tsne_fixed.fit(fixed_features)
# fixed_embedding.embedding_ 中保存了所有固定点的 TSNE 坐标

# ---------------------------
# 处理无人机视图图像（Drone View）并进行外样本映射
# ---------------------------
drone_view_dir = '/data/qiaoq/Project/salad_tz/demo/image_folder'
drone_view_image_paths = sorted(glob.glob(os.path.join(drone_view_dir, '*.jpg')))

# TSNE 结果保存目录
tsne_result_dir = '/data/qiaoq/Project/salad_tz/demo/tsne_result/'
os.makedirs(tsne_result_dir, exist_ok=True)

for drone_path in drone_view_image_paths:
    # 读取并预处理无人机视图图像
    drone_img = Image.open(drone_path).convert('RGB')
    drone_tensor = transform_drone(drone_img).unsqueeze(0).to('cuda')
    with torch.no_grad():
        drone_feature = model(drone_tensor)
    drone_feature = drone_feature.cpu().numpy().flatten()
    
    # 使用 openTSNE 的 transform 方法将新的无人机图像映射到基准 TSNE 空间中
    drone_tsne_coord = fixed_embedding.transform(drone_feature[np.newaxis, :])[0]
    
    # 绘制 TSNE 图：固定点（Ground Truth、Satellite Patch）和无人机视图图像的新映射点
    plt.figure(figsize=(8, 6))
    # 绘制 Ground Truth（绿色，固定）
    plt.scatter(fixed_embedding[0][0], fixed_embedding[0][1],
                color='green', s=100, label='Ground Truth')
    # 绘制卫星 patch（蓝色，固定）
    if len(fixed_embedding) > 1:
        for i in range(1, len(fixed_embedding)):
            if i==1:
                plt.scatter(fixed_embedding[i][0], fixed_embedding[i][1],
                        color='blue', s=40, label='Satellite Patch')
            else:
                plt.scatter(fixed_embedding[i][0], fixed_embedding[i][1],
                        color='blue', s=40)
                
    # 绘制无人机视图图像（红色，动态映射）
    plt.scatter(drone_tsne_coord[0], drone_tsne_coord[1],
                color='red', s=100, label='Drone View')
    
    plt.title("t-SNE Visualization with Fixed Base Points")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(loc="upper right", bbox_to_anchor=(1, 1))
    plt.grid(True)
    
    output_file = os.path.join(
        tsne_result_dir,
        os.path.basename(drone_path).replace('.jpg', '_tsne.jpg')
    )
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"TSNE result saved for {drone_path} as {output_file}")

print("所有无人机视图图像的 TSNE 可视化已完成。")