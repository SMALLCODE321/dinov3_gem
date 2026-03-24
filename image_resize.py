import os
import math
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# --- 配置参数 ---
INPUT_DIR = '/data/xulj/dinov3-salad/sateliteimage/Global_Satellites'  # 原始4096图片路径
OUTPUT_BASE = '/data/xulj/dinov3-salad/datasets/image'    # 切片保存路径
SIZES = [1024, 512, 256]
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

def get_balanced_tasks(input_dir):
    # 获取所有图片并打乱顺序（保证数据分布随机）
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
             if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
    
    total_files = len(files)
    # 比例权重 16:4:1，总份数 21
    unit = total_files // 21
    
    tasks = []
    
    # 分配 1024 尺寸 (约 76.2% 的原图)
    idx_1024 = unit * 16
    for f in files[:idx_1024]:
        tasks.append((f, 1024))
        
    # 分配 512 尺寸 (约 19% 的原图)
    idx_512 = idx_1024 + unit * 4
    for f in files[idx_1024:idx_512]:
        tasks.append((f, 512))
        
    # 分配 256 尺寸 (约 4.8% 的原图)
    for f in files[idx_512:]:
        tasks.append((f, 256))
        
    return tasks

def process_and_slice_full(task):
    img_path, size = task
    try:
        with Image.open(img_path) as img:
            img_name = Path(img_path).stem
            save_dir = os.path.join(OUTPUT_BASE, str(size))
            os.makedirs(save_dir, exist_ok=True)
            
            w, h = img.size
            # 铺满切分逻辑
            cols = w // size
            rows = h // size

            count = 0
            for i in range(cols):
                for j in range(rows):
                    left = i * size
                    top = j * size
                    box = (left, top, left + size, top + size)
                    
                    slice_img = img.crop(box)
                    # 命名：原图名_行_列.jpg
                    slice_img.save(os.path.join(save_dir, f"{img_name}_{i}_{j}.jpg"), quality=95)
                    count += 1
                    
        return f"Success: {img_name} -> {size}px (Total {count} slices)"
    except Exception as e:
        return f"Error: {img_path} - {e}"

def main():
    tasks = get_balanced_tasks(INPUT_DIR)
    
    # 打印预估信息
    print(f"--- 任务分配预估 ---")
    for s in SIZES:
        t_num = sum(1 for t in tasks if t[1] == s)
        slices_per_img = (4096 // s) ** 2
        print(f"尺寸 {s:>4}px: 使用 {t_num:>5} 张原图, 预计产出 {t_num * slices_per_img:>7} 张切片")
    print(f"--------------------")

    # 使用多进程加速
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_and_slice_full, tasks))

    print("\n所有切分任务已完成！")

if __name__ == "__main__":
    main()