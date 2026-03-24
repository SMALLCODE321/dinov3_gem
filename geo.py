import rasterio
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

def plot_point_on_tif(tif_path, lat, lon, patch_size=2000, output_size=(336, 336)):
    """
    在 TIF 卫星图上根据经纬度切片并打点输出
    patch_size: 切取以坐标为中心的卫星图大小 (像素)
    """
    if not os.path.exists(tif_path):
        print(f"❌ 找不到文件: {tif_path}")
        return None

    # 1. 使用 rasterio 获取地理信息并切片
    with rasterio.open(tif_path) as src:
        # 地理坐标转像素坐标 (注意顺序: lon, lat)
        row, col = src.index(lon, lat)
        print(f"📍 目标中心像素坐标: Row={row}, Col={col}")

        # 计算切片窗口 (Window)，确保不越界
        # window = (row_start, col_start, width, height)
        y_start = max(0, int(row - patch_size // 2))
        x_start = max(0, int(col - patch_size // 2))
        
        from rasterio.windows import Window
        window = Window(x_start, y_start, patch_size, patch_size)
        
        # 只读取窗口区域，节省内存
        img_data = src.read([1, 2, 3], window=window)
        img_rgb = img_data.transpose(1, 2, 0)

    # 2. 在切片图上画点
    # 在切片坐标系中的位置
    local_row = int(row - y_start)
    local_col = int(col - x_start)

    # 转换成 BGR 供 OpenCV 画图
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # 画一个醒目的红点 (半径随 patch_size 自动调整)
    radius = int(patch_size * 0.02) 
    cv2.circle(img_bgr, (local_col, local_row), radius=radius, color=(0, 0, 255), thickness=-1)
    cv2.circle(img_bgr, (local_col, local_row), radius=radius+5, color=(255, 255, 255), thickness=3)

    # 3. 准备输出
    final_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(final_rgb)

    # 保存一张直观的图片到本地，方便查看
    vis_save_path = "debug_point_output.jpg"
    pil_img.save(vis_save_path)
    print(f"💾 可视化结果已保存至: {vis_save_path}")

    # 4. 针对 DINOv3 的预处理 (Tensor)
    preprocess = transforms.Compose([
        transforms.Resize(output_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.430, 0.411, 0.296], 
                             std=[0.213, 0.156, 0.143])
    ])
    tensor_img = preprocess(pil_img)

    # 5. 展示
    plt.figure(figsize=(10, 10))
    plt.imshow(final_rgb)
    plt.title(f"Patch {patch_size}x{patch_size} | Center Lat:{lat}, Lon:{lon}")
    plt.axis('off')
    plt.show()

    return tensor_img

# --- 执行 ---
if __name__ == "__main__":
    TIF_FILE = "/data/xulj/dinov3-salad/datasets/UAV_VisLoc_dataset/01/satellite01.tif"
    
    # 你的目标经纬度
    target_lat = 29.76096029 
    target_lon = 115.9747973
    
    # 执行：切出 2000x2000 的图并打点，最后缩放到 336 给模型
    processed_tensor = plot_point_on_tif(
        TIF_FILE, 
        target_lat, 
        target_lon, 
        patch_size=2000, 
        output_size=(336, 336)
    )
    
    if processed_tensor is not None:
        print("✅ Tensor 形状:", processed_tensor.shape)