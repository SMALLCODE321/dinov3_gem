import rasterio
from rasterio.windows import Window
import cv2
import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def process_large_satellite_patches(root_dir, patch_size=1024, overlap=0.25):
    """
    patch_size: 设置为 1024 或更大，确保覆盖范围广于无人机
    overlap: 覆盖率可以适当降低，因为切片变大了
    """
    scenarios = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.isdigit()]
    scenarios.sort()

    for sn in scenarios:
        tif_path = os.path.join(root_dir, sn, f"satellite{sn}.tif")
        if not os.path.exists(tif_path): continue

        # 专门存放在大尺寸文件夹下
        output_dir = os.path.join(root_dir, sn, f"gallery_large_{patch_size}")
        os.makedirs(output_dir, exist_ok=True)
        
        with rasterio.open(tif_path) as src:
            H, W = src.height, src.width
            # 计算步长：大的切片可以用大的步长，减少冗余
            stride = int(patch_size * (1 - overlap))
            
            y_steps = range(0, H - patch_size + 1, stride)
            x_steps = range(0, W - patch_size + 1, stride)
            
            index_records = []
            
            print(f"\n🌍 场景 {sn}: 正在生成大范围卫星切片 ({patch_size}x{patch_size})")
            
            with ThreadPoolExecutor(max_workers=8) as executor:
                for y in tqdm(y_steps, desc=f"Row processing {sn}"):
                    for x in x_steps:
                        # 1. 读取大范围窗口
                        window = Window(x, y, patch_size, patch_size)
                        patch_data = src.read([1, 2, 3], window=window)
                        patch_rgb = patch_data.transpose(1, 2, 0)
                        
                        # 2. 坐标转换 (依然记录中心点)
                        center_x, center_y = x + patch_size/2, y + patch_size/2
                        lon, lat = src.xy(center_y, center_x)
                        
                        file_name = f"sat_large_{sn}_y{y}_x{x}.jpg"
                        save_path = os.path.join(output_dir, file_name)
                        
                        # 直接保存 1024x1024 的原图，不 Resize
                        executor.submit(cv2.imwrite, save_path, cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR))
                        
                        index_records.append({
                            "image_name": file_name,
                            "center_lat": lat,
                            "center_lon": lon,
                            "patch_size_pixel": patch_size
                        })

            pd.DataFrame(index_records).to_csv(os.path.join(output_dir, "index_info.csv"), index=False)

if __name__ == "__main__":
    DATASET_ROOT = "/data/xulj/dinov3-salad/datasets/UAV_VisLoc_dataset"
    # 这里切 1024，物理覆盖范围是 512 的四倍
    process_large_satellite_patches(root_dir=DATASET_ROOT, patch_size=2000)