import argparse
from tqdm import tqdm
import pandas as pd

def parse_args():
    """
    解析命令行参数，用于设置文件路径、裁剪尺寸和步长信息。
    
    注意：
    - --crop_size 需要传入两个整数，第一个表示裁剪块的高度，第二个表示裁剪块的宽度，
      例如: --crop_size 1080 2048
    - --step_size 需要传入两个整数，第一个表示纵向移动的步长，第二个表示横向移动的步长，
      例如: --step_size 864 1638
    """
    parser = argparse.ArgumentParser(
        description="创建图像块索引（矩形切块）",  
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 基础信息文件（包含图像名称、图像高度和宽度）
    parser.add_argument(
        "--base_info_file", 
        type=str, 
        default='./data/base_map_info.txt', 
        help="图像基础信息文件路径，文件中每一行包含图像名称、图像高度和宽度"
    )
    # 裁剪尺寸参数（高度和宽度）
    parser.add_argument(
        "--crop_size", 
        nargs='+', 
        type=lambda s: tuple(map(int, s.split(','))), 
        required=True, 
        help="裁剪尺寸列表，依次为裁剪块的高度和宽度，例如: --crop_size 1080,2048"
    )
    # 裁剪步长参数（纵向步长和横向步长）
    parser.add_argument(
        "--step_size", 
        nargs='+', 
        type=lambda s: tuple(map(int, s.split(','))), 
        required=True,
        help="裁剪时的步长列表，依次为纵向和横向的步长，例如: --step_size 864,1638"
    )
    # 输出索引信息的CSV文件路径
    parser.add_argument(
        "--index_info_file", 
        type=str, 
        required=True, 
        help="保存索引信息的CSV文件路径, 每个条目包含一个裁剪块的信息"
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__': 
    args = parse_args()
    
    base_info_file = args.base_info_file
    crop_h, crop_w = args.crop_size[0][0], args.crop_size[0][1]
    step_h, step_w = args.step_size[0][0], args.step_size[0][1]

    # 用于存储所有生成的图像块信息
    patches = []   
    
    # 读取基础信息文件，每行包含图像文件名和它的尺寸信息
    with open(base_info_file, 'r') as file: 
        for line in tqdm(file, desc="处理图像信息"):
            # 去除首尾空白字符并按逗号分割
            parts = line.strip().split(',') 
            # 解包图像信息：文件名、图像高度(H)和宽度(W)
            image_name, W, H = parts[0], int(parts[1]), int(parts[2])
            
            # 遍历所有可能的裁剪起点（纵向和横向移动）
            for top in range(0, H, step_h):
                for left in range(0, W, step_w):
                    # 计算裁剪块的右下角位置，防止越界
                    bottom = min(top + crop_h, H)
                    right = min(left + crop_w, W)
                    
                    # 若裁剪高度不足，则调整上边界
                    if bottom - top < crop_h:
                        top_adj = max(bottom - crop_h, 0)
                    else:
                        top_adj = top
                    # 若裁剪宽度不足，则调整左边界
                    if right - left < crop_w:
                        left_adj = max(right - crop_w, 0)
                    else:
                        left_adj = left
                    
                    # 构造当前裁剪块的信息字典
                    patch = {
                        "image_name": image_name, 
                        "x1": left_adj, 
                        "y1": top_adj, 
                        "x2": right, 
                        "y2": bottom
                    } 
                    patches.append(patch)   
        
    # 将所有裁剪块信息转换成DataFrame，并保存为CSV文件
    df = pd.DataFrame(patches)
    df.to_csv(args.index_info_file, index=False)