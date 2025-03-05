import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import os
import cv2
import glob 
import faiss
import pandas as pd
import sys
from vpr_model import VPRModel 
# Dataloader
from dataloaders.TestDataset import TestListDataset
from calculate_iou import file_IoU_calculate
from match_image_points import PointMatcher

def get_descriptors(model, dataloader, device):
    descriptors = []
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for batch in tqdm(dataloader, 'Calculating descritptors...'):
                imgs = batch 
                output = model(imgs.to(device)).cpu()
                descriptors.append(output)
    
    return torch.cat(descriptors)

def load_model(ckpt_path):
    model = VPRModel(
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
    )

    loaded_state_dict = torch.load(args.ckpt_path)
    if "state_dict" in loaded_state_dict.keys():
        loaded_state_dict = loaded_state_dict["state_dict"]
    model.load_state_dict(loaded_state_dict)
    model = model.eval()
    model = model.to('cuda')
    print(f"Loaded model from {ckpt_path} Successfully!")
    return model

"""
提取图像中的边缘点，并返回排序后的边缘点的起始和结束坐标

输入：image：输入的图像，通常是一个彩色图像

输出：result：一个包含两个坐标的数组，分别是排序后的边缘点的起始和结束坐标
""" 
def get_trans_points(image):
    #图像转化为灰度图
    image = np.array(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image[gray_image>0]=255

     #创建扩展的黑色边缘图像
    black_edge = np.zeros((gray_image.shape[0] + 2, gray_image.shape[1] + 2), dtype=np.uint8) #创建一个比原始灰度图像大2像素的黑色边缘图像

    black_edge[1:-1, 1:-1] = gray_image #将原始灰度图像复制到黑色边缘图像的中心

    # 定义一个结构元素，这里使用3x3的矩形  
    kernel = np.ones((3, 3), np.uint8)
    
    # 应用腐蚀操作
    eroded_image = cv2.erode(black_edge, kernel, iterations=1)

    #去除腐蚀后图像的扩展边缘，恢复到原始图像大小
    eroded_image = eroded_image[1:-1, 1:-1]

    eroded_image=gray_image-eroded_image
    """
    获取边缘点
    1.通过原始灰度图像减去腐蚀后的图像，得到边缘点
    2.使用np.where找到值为255的像素坐标
    3.将坐标转换为列表并转换为numpy数组
    """
    rows, cols = np.where(eroded_image == 255)
    white_pixel_coords = list(zip(rows, cols))
    edge = np.array(white_pixel_coords)
    # print("边缘点矩阵shape", edge.shape)
    edge = edge[:, [1, 0]]
    sorted_edge = np.array(sorted(edge, key=lambda x: (x[0], x[1])))
    result = np.array([sorted_edge[0],sorted_edge[-1]])
    return result


"""
这个函数transform_points的目的是使用给定的变换矩阵H对一组二维点进行变换
这种变换通常用于图像处理中的坐标变换，比如仿射变换或透视变换

输入：
points：一个二维点的数组，形状为(N, 2)，其中N是点的数量
H：一个变换矩阵，形状为(3, 3)

输出：
变换后的点坐标数组，形状为(N, 2)
"""
def transform_points(points, H): 
    #创建一个与点数量相同的列向量，所有元素为1
    ones = np.ones((points.shape[0], 1))
    #将这个列向量与原始点坐标水平堆叠，形成齐次坐标。
    #齐次坐标是用于图形变换的一种表示方法，它通过增加一个额外的坐标（通常是1）来简化变换矩阵的计算
    points_homogeneous = np.hstack([points, ones])
    """
    使用变换矩阵H与齐次坐标矩阵进行矩阵乘法
    @符号表示矩阵乘法
    T表示矩阵的转置
    变换后的点坐标仍然是齐次坐标
    """
    transformed_points = (H @ points_homogeneous.T).T
    transformed_points /= transformed_points[:, 2][:, np.newaxis]
    transformed_points = np.round(transformed_points)
    
    return np.asarray(transformed_points[:, :2]) 

def extend_to_square_within_image(x1, y1, x2, y2, image_width, image_height, target_size=2000): 
    # Calculate the original center
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Calculate the desired half size for the square
    half_target_size = target_size / 2

    # Calculate potential new coordinates
    new_x1 = center_x - half_target_size
    new_y1 = center_y - half_target_size
    new_x2 = center_x + half_target_size
    new_y2 = center_y + half_target_size

    # Ensure new coordinates don't exceed image boundaries
    if new_x1 < 0:
        new_x1 = 0
        new_x2 = min(target_size, image_width)

    if new_y1 < 0:
        new_y1 = 0
        new_y2 = min(target_size, image_height)

    if new_x2 > image_width:
        new_x2 = image_width
        new_x1 = max(0, image_width - target_size)

    if new_y2 > image_height:
        new_y2 = image_height
        new_y1 = max(0, image_height - target_size)

    return int(new_x1), int(new_y1), int(new_x2), int(new_y2)
  
def compute_localization(query_im, query_res, index_info_csv, db_dir, matcher, extend_size=2000):
    
    result_corr = []
    find_flag = False 
    max_match_points = 0
     
    i=0           
    for ind in query_res:
        ind_data = index_info_csv.iloc[[ind]] 
        # 找到对应大图
        match_im_name = os.path.join(db_dir, ind_data['image_name'].values[0])
        match_im = cv2.imread(match_im_name, cv2.IMREAD_COLOR)
        patch_start_x = ind_data['x1'].values[0]
        patch_start_y = ind_data['y1'].values[0]
        patch_end_x = ind_data['x2'].values[0]
        patch_end_y = ind_data['y2'].values[0] 
        
        match_start_x, match_start_y, match_end_x, match_end_y = extend_to_square_within_image(patch_start_x, patch_start_y, 
                                                                                            patch_end_x, patch_end_y,
                                                    match_im.shape[1], match_im.shape[0], extend_size) 
        
        crop_corr = (match_start_x, match_start_y, match_end_x, match_end_y) 
        # 在大图中切出用于匹配的的图 
        match_im_crop = match_im[match_start_y:match_end_y, match_start_x:match_end_x, :]
        # match_im_crop_i = match_im_i  
        test_pts, match_pts = [], []
         
             
        test_pts, match_pts = matcher.match_images(query_im, match_im_crop)  
        print(type(test_pts))
        print(test_pts)  
        if len(test_pts) > max_match_points:
            # 从前往后找，如果下一个的匹配点数没有前面的点数多，那么跳过    
            max_match_points = len(test_pts)
            # 获取test图片中用于仿射变换的点的坐标 
            if len(test_pts) > 10: 
                test_pts_np = np.asarray(test_pts)
                match_pts_np = np.asarray(match_pts)
                H, mask = cv2.findHomography(test_pts_np, match_pts_np, cv2.RANSAC, 5.0)
                
                if H is not None:  
                    trans_points = get_trans_points(query_im) 
                    # print("test图角点\n")
                    # print(trans_points)  
                    tar_points = transform_points(trans_points, H)
                    
                    results_start_x = int(min(tar_points[0][0], tar_points[1][0]))
                    results_end_x = int(max(tar_points[0][0], tar_points[1][0]))
                    results_start_y = int(min(tar_points[0][1], tar_points[1][1]))
                    results_end_y = int(max(tar_points[0][1], tar_points[1][1]))

                    if results_start_x < 0: results_start_x = 0
                    if results_start_y < 0: results_start_y = 0
                    if results_end_x > match_im_crop.shape[1]: results_end_x = match_im_crop.shape[1]
                    if results_end_y > match_im_crop.shape[0]: results_end_y = match_im_crop.shape[0]
                        
                    if (results_end_x - results_start_x) * (results_end_y - results_start_y) > 0:  
                        
                        results_start_x = int(results_start_x)
                        results_end_x = int(results_end_x)
                        results_start_y = int(results_start_y)
                        results_end_y = int(results_end_y) 
                        
                        final_start_x = crop_corr[0] + results_start_x
                        final_start_y = crop_corr[1] + results_start_y
                        final_end_x = crop_corr[0] + results_end_x
                        final_end_y = crop_corr[1] + results_end_y
                        result_corr = [final_start_x, final_start_y, final_end_x, final_end_y]
                        
                        find_flag = True
                        break
                  
    return find_flag, result_corr, match_im_name

def parse_args():
    parser = argparse.ArgumentParser(
        # description="Eval VPR model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Model parameters
    parser.add_argument("--input", type=str, default='./data/test_images', help="Path to the image directory,\
                        or can be a single image file path")
    parser.add_argument("--ext", nargs='+', type=str, default='.jpg', help="Ext of the query images") 
    parser.add_argument('--output_file', type=str, default='./RESULT.txt', help="Path to the save the results") 
    parser.add_argument('--gt_file', type=str, default=None, help="Path to the save the results") 
    parser.add_argument("--db_dir", type=str, default='./data/base_map', help="Path to the database image directory")
    parser.add_argument("--ckpt_path", type=str, default='./dino_salad.ckpt', help="Path to the checkpoint") 
    parser.add_argument("--index_path", type=str, default='./index.faiss', help="Path to the index") 
    parser.add_argument("--index_info_csv", type=str, default='./index_info.csv', help="Path to the crop image file")
    parser.add_argument('--image_size', type=int, default=None, help='Image size for the global feature extractor') 
    parser.add_argument('--extend_size', type=int, default=1500, help='Extend size for matching') 
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size') 
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers')
    parser.add_argument('--matcher_type', type=str, default='flann', help='Matcher type(flann or adalam)')
    parser.add_argument('--topk_index', type=int, default=10, help='Top K returned by index')

    args = parser.parse_args()
 
    return args  


if __name__ == '__main__':
    #设置PyTorch的CUDA环境
    torch.backends.cudnn.benchmark = True 
    torch.backends.cudnn.deterministic = True
    
    #解析命令行参数
    args = parse_args()
    print(args)

    #加载模型 
    model = load_model(args.ckpt_path) 
    
    #获取测试图像列表
    test_fnames = []  
    if os.path.isdir(args.input):
        for _e in args.ext: 
            test_fnames += glob.glob(os.path.join(args.input,'*'+_e))  
        test_fnames = sorted(test_fnames)
    else:
        # Input of args.input is a specific image path
        test_fnames =[args.input]  

    #创建测试数据集和dataloader      
    test_dataset = TestListDataset(im_list=test_fnames, image_size=args.image_size)
    data_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size,
                                shuffle=False, drop_last=False, pin_memory=True) 

    #提取图像描述符  
    descriptors = get_descriptors(model, data_loader, 'cuda')
    descriptors = descriptors[:len(test_fnames)].numpy()
    
    #释放内存
    del model
    torch.cuda.empty_cache()

    #加载索引和索引信息
    print(args.index_path)
    index = faiss.read_index(args.index_path) 
    index_info_csv = pd.read_csv(args.index_info_csv) # Load the CSV data into a DataFrame 
    
    #使用FAISS索引搜索最相似的图像，并重置索引以释放内存
    distances, indices = index.search(descriptors, args.topk_index)
    index.reset()
    del index
    # matcher = [
    #         PointMatcher(feature_type='sift', matcher_type='flann'),
    #         # PointMatcher(feature_type='disk', matcher_type='lightglue'),
    #         #    PointMatcher(feature_type='orb', matcher_type='bf'),
    #         #    PointMatcher(feature_type='xfeat', matcher_type=args.matcher_type),
    #            ]

    #初始化一个点匹配器，用于后续的图像匹配
    matcher = PointMatcher(feature_type='sift', matcher_type=args.matcher_type)
    
    """
    对于每个测试图像，读取图像，执行定位计算，并将结果写入输出文件
    如果找到匹配，则记录匹配的图像名和定位结果；
    如果没有找到，则记录默认值
    将每个测试图像的top-k匹配图像名称存储在topk_dic字典中
    """
    topk_dic = {}  
    with open(args.output_file, 'w') as result_writer: 
        
        for i in tqdm(range(len(test_fnames))): 
            print(test_fnames[i])
            query_im = cv2.imread(test_fnames[i], cv2.IMREAD_COLOR)  
            
            find_flag =False
            find_flag, result, match_im_name = compute_localization(query_im, indices[i], index_info_csv, args.db_dir, matcher, args.extend_size)
            
            if find_flag:
                 print(find_flag, result, match_im_name)
                 result_writer.write(os.path.basename(test_fnames[i]) + ' ' + os.path.basename(match_im_name) + \
                                    ' ' + str(result[0]) + ' ' + str(result[1]) + ' ' + str(result[2]) + \
                                    ' ' + str(result[3]) + '\n')
            else: 
                final_start_x = 0
                final_start_y = 0
                final_end_x = 5000
                final_end_y = 5000
                match_im_name = index_info_csv.iloc[[indices[i][0]]]['image_name'].values[0]
                result_writer.write(os.path.basename(test_fnames[i]) + ' ' + os.path.basename(match_im_name) + ' ' + \
                                                     str(final_start_x) + ' ' + str(final_start_y) + ' ' + str(final_end_x) +\
                                                         ' ' + str(final_end_y) + '\n')
                
            topk_dic[os.path.basename(test_fnames[i])] = [index_info_csv.iloc[[ind]]['image_name'].values[0] for ind in indices[i]]