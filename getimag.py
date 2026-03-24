import os
import shutil
from tqdm import tqdm

def extract_drone_150m_only(root_path, output_path):
    """
    root_path: SUES-200-512x512 的根目录
    output_path: 提取后的存放路径
    """
    # 定位无人机图根目录
    drone_root = os.path.join(root_path, 'drone_view_512')
    
    # 定义目标存放目录
    q_150_dir = os.path.join(output_path, 'sues200_drone_250m')
    os.makedirs(q_150_dir, exist_ok=True)

    if not os.path.exists(drone_root):
        print(f"错误: 找不到目录 {drone_root}")
        return

    # 获取所有类别文件夹 (0001, 0002...)
    class_folders = [f for f in os.listdir(drone_root) if os.path.isdir(os.path.join(drone_root, f))]
    
    print(f"正在从 {len(class_folders)} 个类别中提取 150m 无人机图像...")
    
    count = 0
    for cls in tqdm(class_folders):
        # 对应路径: drone_view_512/0001/150/
        src_150_path = os.path.join(drone_root, cls, '250')
        
        if os.path.exists(src_150_path):
            # 在目标路径保留类别子目录，防止文件名冲突，且方便评估
            dst_cls_dir = os.path.join(q_150_dir, cls)
            os.makedirs(dst_cls_dir, exist_ok=True)
            
            for img in os.listdir(src_150_path):
                # 只复制图片文件
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    shutil.copy(
                        os.path.join(src_150_path, img), 
                        os.path.join(dst_cls_dir, img)
                    )
                    count += 1

    print(f"\n✅ 提取完成！共提取 {count} 张图像。")
    print(f"提取结果存放在: {q_150_dir}")

if __name__ == "__main__":
    # 设定你的路径
    ORIGIN = "/data/xulj/dinov3-salad/SUES-200-512x512"
    TARGET = "/data/xulj/dinov3-salad/SUES-200-512x512"
    extract_drone_150m_only(ORIGIN, TARGET)