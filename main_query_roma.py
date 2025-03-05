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
from vpr_model import VPRModel 
# Dataloader
from dataloaders.TestDataset import TestListDataset
from calculate_iou import file_IoU_calculate
from matching import get_matcher 
import shutil
import uuid
 
 
TEM_DIR = './temp'
MIN_AREA = 200 * 200
MAX_AREA = 1500 * 1500  

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

    loaded_state_dict = torch.load(args.ckpt_path, map_location=torch.device('cuda:0'))
    if "state_dict" in loaded_state_dict.keys():
        loaded_state_dict = loaded_state_dict["state_dict"]
    model.load_state_dict(loaded_state_dict)
    model = model.eval()
    model = model.to('cuda')
    print(f"Loaded model from {ckpt_path} Successfully!")
    return model

def get_trans_points(image):
    image = np.array(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image[gray_image>0]=255

    black_edge = np.zeros((gray_image.shape[0] + 2, gray_image.shape[1] + 2), dtype=np.uint8)

    black_edge[1:-1, 1:-1] = gray_image

    # 定义一个结构元素，这里使用3x3的矩形  
    kernel = np.ones((3, 3), np.uint8)
  
    # 应用腐蚀操作
    eroded_image = cv2.erode(black_edge, kernel, iterations=1)

    eroded_image = eroded_image[1:-1, 1:-1]

    eroded_image=gray_image-eroded_image
    cv2.imwrite('eroded_image.jpg', eroded_image)
    rows, cols = np.where(eroded_image == 255)
    white_pixel_coords = list(zip(rows, cols))
    edge = np.array(white_pixel_coords)
    # print("边缘点矩阵shape", edge.shape)
    edge = edge[:, [1, 0]]
    sorted_edge1 = np.array(sorted(edge, key=lambda x: (x[0], x[1])))

    sort_edge_y = np.argsort(edge[:, 1])
    sort_edge_x = sort_edge_y.copy()
    unique_y, counts = np.unique(edge[:, 1], return_counts=True)
    for y, count in zip(unique_y, counts):
        indices = np.where(edge[sort_edge_y, 1] == y)[0]
        sort_edge_x[indices] = sort_edge_y[indices][np.argsort(edge[sort_edge_y, 0][indices])[::-1]]
    sorted_edge2 = edge[sort_edge_x]

    result = np.array([sorted_edge1[0],sorted_edge2[0], sorted_edge2[-1], sorted_edge1[-1]])
    return result  
  

def inverse_rotate_point(x, y, width, height, angle):
    """
    Reverse the rotation of a point (x, y) by a specific angle.

    Parameters:
    - x, y: Coordinates of the point.
    - width, height: Width and height of the image.
    - angle: Rotation angle (0, 90, 180, 270 degrees).

    Returns:
    - New (x', y') coordinates after applying the inverse rotation.
    """
    if angle == 0:
        return x, y
    elif angle == 90:
        return height - y, x
    elif angle == 180:
        return width - x, height - y
    elif angle == 270:
        return y, width - x
    else:
        raise ValueError("Angle must be one of [0, 90, 180, 270]")
      
def transform_points(points, H): 
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack([points, ones])
    transformed_points = (H @ points_homogeneous.T).T
    transformed_points /= transformed_points[:, 2][:, np.newaxis]
    transformed_points = np.round(transformed_points)
    
    return np.asarray(transformed_points[:, :2]) 
  
def extend_to_square_within_image(points, im_size, target_size=2000): 
    # Calculate the original center
    x1, y1, x2, y2 = points[0], points[1], points[2], points[3]
    image_width, image_height = im_size
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

def expand_box(points, im_size, expansion_size=200):
    # Extract the original points and image dimensions
    x1, y1, x2, y2 = points[0], points[1], points[2], points[3]
    image_width, image_height = im_size

    # Expand the coordinates by the specified expansion size
    new_x1 = x1 - expansion_size
    new_y1 = y1 - expansion_size
    new_x2 = x2 + expansion_size
    new_y2 = y2 + expansion_size

    # Ensure the new coordinates stay within image boundaries
    new_x1 = max(new_x1, 0)
    new_y1 = max(new_y1, 0)
    new_x2 = min(new_x2, image_width)
    new_y2 = min(new_y2, image_height)

    # Calculate the width and height of the new expanded box
    box_width = new_x2 - new_x1
    box_height = new_y2 - new_y1

    # Ensure the box is a square
    if box_width > box_height:
        # Adjust the height to match the width
        diff = box_width - box_height
        new_y1 = max(new_y1 - diff // 2, 0)
        new_y2 = min(new_y2 + diff // 2 + (diff % 2), image_height)
    elif box_height > box_width:
        # Adjust the width to match the height
        diff = box_height - box_width
        new_x1 = max(new_x1 - diff // 2, 0)
        new_x2 = min(new_x2 + diff // 2 + (diff % 2), image_width)

    # Return the new square coordinates, ensuring they stay within boundaries
    new_x1 = max(new_x1, 0)
    new_y1 = max(new_y1, 0)
    new_x2 = min(new_x2, image_width)
    new_y2 = min(new_y2, image_height)

    return int(new_x1), int(new_y1), int(new_x2), int(new_y2)
 
  
def compute_homography(img0, img1): 
    result = matcher(img0, img1)  
    num_inliers, H = result['num_inliers'], result['H']
    return num_inliers, H

def compute_best_homography(img0, db_im, matcher, matcher_size):
    best_H = None
    best_num_inliers = 0
    best_angle = 0
    rotation_angles = [0, 90, 180, 270]
    
    for angle in rotation_angles:
        cv2.imwrite('db_im.jpg', db_im)
        img1 = matcher.load_image('db_im.jpg', resize=matcher_size, rot_angle=angle)
        num_inliers, H = compute_homography(img0, img1)
        
        if num_inliers > best_num_inliers:
            best_H = H
            best_num_inliers = num_inliers
            best_angle = angle
            
    return best_num_inliers, best_H, best_angle

def compute_coordinates(trans_points, H, matcher_size, query_im_size, match_im_size, ori_crop_corr, inverse_angle=0):
    
    find_flag = False
    S1 = np.array([[matcher_size / query_im_size[0], 0, 0], 
                   [0, matcher_size / query_im_size[1], 0], 
                   [0, 0, 1]])

    S2 = np.array([[match_im_size[0] / matcher_size, 0, 0], 
                   [0, match_im_size[1] / matcher_size, 0], 
                   [0, 0, 1]])
    
    H_adjusted = S2 @ H @ S1
                
    tar_points = transform_points(trans_points, H_adjusted)
        
    # Rotate points back to the original coordinate system
    tar_points[0] = inverse_rotate_point(tar_points[0][0], tar_points[0][1], match_im_size[0], match_im_size[1], inverse_angle)
    tar_points[1] = inverse_rotate_point(tar_points[1][0], tar_points[1][1], match_im_size[0], match_im_size[1], inverse_angle)
    tar_points[2] = inverse_rotate_point(tar_points[2][0], tar_points[2][1], match_im_size[0], match_im_size[1], inverse_angle)
    tar_points[3] = inverse_rotate_point(tar_points[3][0], tar_points[3][1], match_im_size[0], match_im_size[1], inverse_angle)
    
    # Compute the bounding box from the transformed points
    results_start_x = int(np.min(tar_points[:, 0]))
    results_end_x = int(np.max(tar_points[:, 0]))
    results_start_y = int(np.min(tar_points[:, 1]))
    results_end_y = int(np.max(tar_points[:, 1]))
 
    # Ensure coordinates are within the image boundaries
    results_start_x = max(0, results_start_x)
    results_start_y = max(0, results_start_y)
    results_end_x = min(match_im_size[0], results_end_x)
    results_end_y = min(match_im_size[1], results_end_y)

    # Ensure that the start coordinates are less than the end coordinates
    results_start_x = min(results_start_x, results_end_x)
    results_start_y = min(results_start_y, results_end_y)

    # Ensure the final coordinates are still within the image boundaries
    results_end_x = min(results_end_x, match_im_size[0])
    results_end_y = min(results_end_y, match_im_size[1])
    
        
    result_area = (results_end_x - results_start_x) * (results_end_y - results_start_y)
     
    if result_area > MIN_AREA:   
        final_start_x = ori_crop_corr[0] + results_start_x
        final_start_y = ori_crop_corr[1] + results_start_y
        final_end_x = ori_crop_corr[0] + results_end_x
        final_end_y = ori_crop_corr[1] + results_end_y
        result_corr = [final_start_x, final_start_y, final_end_x, final_end_y]
        
        find_flag = True  
        
        return find_flag, result_corr  
    
    return find_flag, None
               
def match(query_im, db_im, matcher, matcher_size, rot_angle=0, apply_rot_aug=False):  
          
    if apply_rot_aug:
        num_inliers, H, rot_angle = compute_best_homography(query_im, db_im, matcher, matcher_size)  
        
    else: 
        cv2.imwrite('db_im.jpg', db_im)
        img1 = matcher.load_image('db_im.jpg', resize=matcher_size, rot_angle=rot_angle)  
        num_inliers, H = compute_homography(query_im, img1) 
    
    return H, num_inliers, rot_angle 
      
    
def compute_localization(args, query_im, query_res, index_info_csv, matcher):
    
    result_corr = []
    find_flag = False  
    trans_points = get_trans_points(query_im)
    cv2.imwrite('img0.jpg', query_im)   
    img0 = matcher.load_image('img0.jpg', resize=args.matcher_size) 
    temp_cal = 1
    for ind in query_res:
        ind_data = index_info_csv.iloc[[ind]] 
        # Find the corresponding large image
        match_im_name = os.path.join(args.db_dir, ind_data['image_name'].values[0])
        match_im = cv2.imread(match_im_name, cv2.IMREAD_COLOR)
        patch_start_x = ind_data['x1'].values[0]
        patch_start_y = ind_data['y1'].values[0]
        patch_end_x = ind_data['x2'].values[0]
        patch_end_y = ind_data['y2'].values[0] 
        patch_points = (patch_start_x, patch_start_y,  patch_end_x, patch_end_y)
        print(patch_points)
        print(patch_start_y, patch_end_y, patch_start_x, patch_end_x)
        patch_img = match_im[patch_start_x:patch_end_x, patch_start_y:patch_end_y]
        print(patch_img.shape)
        cv2.imwrite('./result/' + str(temp_cal) + '.jpg', patch_img)
        # print(match_im_name, patch_points)
        # cv2.imwrite('./result/patchimg' + str(temp_cal) + '.jpg', patch_img)
        temp_cal = temp_cal + 1

    ####################changed
    #     query_im_size = (query_im.shape[1], query_im.shape[0])
    #     H, num_inliers, rot_angle, crop_corr = None, 0, 0, patch_points
        
    #     args.extend_size = min((patch_end_x-patch_start_x)*3, 2000)
        
    #     if isinstance(args.extend_size, list):  
    #         extend_size = args.extend_size[0]
            
    #         for extend_size_i in args.extend_size:
                
    #             crop_corr_tmp = extend_to_square_within_image(patch_points,(match_im.shape[1], match_im.shape[0]), extend_size_i) 
    #             # Crop the matching image from the large image
    #             match_im_crop = match_im[crop_corr_tmp[1]:crop_corr_tmp[3], crop_corr_tmp[0]:crop_corr_tmp[2], :]  
    #             match_im_size_tmp = (match_im_crop.shape[1], match_im_crop.shape[0])
                
    #             H_tmp, num_inliers_tmp, rot_angle_tmp = match(img0, match_im_crop, matcher, args.matcher_size, apply_rot_aug= args.apply_rot_aug)
                
    #             if H_tmp is not None and num_inliers_tmp > num_inliers:
    #                H, num_inliers, rot_angle =  H_tmp, num_inliers_tmp, rot_angle_tmp
    #                match_im_size = match_im_size_tmp
    #                crop_corr = crop_corr_tmp
    #                extend_size = extend_size_i
                   
    #     else: 
    #         extend_size = args.extend_size 
    #         crop_corr = extend_to_square_within_image(patch_points, (match_im.shape[1], match_im.shape[0]), extend_size)
    #         # Crop the matching image from the large image
    #         match_im_crop = match_im[crop_corr[1]:crop_corr[3], crop_corr[0]:crop_corr[2], :]  
    #         match_im_size = (match_im_crop.shape[1], match_im_crop.shape[0])
    #         H, num_inliers, rot_angle = match(img0, match_im_crop, matcher, args.matcher_size, apply_rot_aug= args.apply_rot_aug)
        
    #     if num_inliers > args.inlier_threshold:   
            
    #         if H is not None:
    #             find_flag, result_corr = compute_coordinates(trans_points, H, args.matcher_size, query_im_size, match_im_size, crop_corr, rot_angle)
                 
    #             if args.refine and find_flag: 
    #                 # refine_expand_size = 0.5 * max(result_corr[3]-result_corr[1], result_corr[2]-result_corr[0]) 
    #                 # result_corr_new =  expand_box(result_corr, (match_im.shape[1], match_im.shape[0]), refine_expand_size)   
    #                 result_corr_new = extend_to_square_within_image(result_corr, (match_im.shape[1], match_im.shape[0]), extend_size) 
                    
    #                 match_im_crop = match_im[result_corr_new[1]:result_corr_new[3], result_corr_new[0]:result_corr_new[2], :]
                    
    #                 match_im_size = (match_im_crop.shape[1], match_im_crop.shape[0])
                    
    #                 H_refine, num_inliers_refine, _ = match(img0, match_im_crop, matcher, args.matcher_size, rot_angle=rot_angle)
                    
    #                 if num_inliers_refine > args.inlier_threshold:   
    #                     if H_refine is not None:
    #                         find_flag_refine, result_corr_refine = compute_coordinates(trans_points, H_refine, args.matcher_size, query_im_size,
    #                                                                             match_im_size, result_corr_new, inverse_angle=rot_angle)
                        
    #                         if find_flag_refine:
    #                             find_flag, result_corr, num_inliers = find_flag_refine, result_corr_refine, num_inliers_refine
                         
                            
    #             if find_flag:        
    #                 return find_flag, result_corr, match_im_name 
              
    # # If no matches are found
    # return find_flag, result_corr, None
 

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
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers')
    parser.add_argument('--matcher', type=str, default='sift-lg', help='Matcher type(flann or adalam)')
    parser.add_argument('--num_kp', type=int, default=4096, help='Matcher type(flann or adalam)')
    parser.add_argument('--matcher_size', type=int, default=1024, help='Matcher resize ')
    # parser.add_argument('--matcher2', type=str, default='sift-lg', help='Matcher type(flann or adalam)')
    # parser.add_argument('--num_kp2', type=int, default=4096, help='Matcher type(flann or adalam)')
    # parser.add_argument('--matcher_size2', type=int, default=768, help='Matcher resize ')
    parser.add_argument('--topk_index', type=int, default=10, help='Top K returned by index')
    parser.add_argument('--extend_size', nargs='+', type=int, help='Extend size for matching') 
    parser.add_argument('--apply_rot_aug', action='store_true', help='Apply rotation augmentation if specified')
    parser.add_argument('--inlier_threshold', type=int, default=4, help='inlier_threshold')
    parser.add_argument('--refine_expand_size', type=int, default=200, help='refine_expand_size')
    parser.add_argument('--ransac_reproj_thresh', type=float, default=5.0, help='ransac_reproj_thresh')
    parser.add_argument('--ransac_iters', type=int, default=2000, help='ransac_iters')
    parser.add_argument('--ransac_conf', type=float, default=0.95, help='ransac_conf') 
    parser.add_argument('--refine',  action='store_true',  help='whether perform the refinement') 
    args = parser.parse_args()
 
    return args


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True 
    torch.backends.cudnn.deterministic = True
    args = parse_args()
    print(args)
     
    model = load_model(args.ckpt_path) 
     
    test_fnames = []  
    if os.path.isdir(args.input):
        for _e in args.ext: 
            test_fnames += glob.glob(os.path.join(args.input,'*'+_e))  
        test_fnames = sorted(test_fnames)
    else:
        # Input of args.input is a specific image path
        test_fnames =[args.input]  
          
    test_dataset = TestListDataset(im_list=test_fnames, image_size=args.image_size)
    data_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size,
                                shuffle=False, drop_last=False, pin_memory=True) 
      
    descriptors = get_descriptors(model, data_loader, 'cuda')
    descriptors = descriptors[:len(test_fnames)].numpy()
    
    del model
    torch.cuda.empty_cache()
    
    index = faiss.read_index(args.index_path) 
    index_info_csv = pd.read_csv(args.index_info_csv) # Load the CSV data into a DataFrame 
       
    distances, indices = index.search(descriptors, args.topk_index)
    index.reset()
    del index
     
    matcher = get_matcher(args.matcher, device='cuda', max_num_keypoints=args.num_kp, 
                          ransac_reproj_thresh=args.ransac_reproj_thresh, ransac_iters=args.ransac_iters)  
    
    # matcher2 = get_matcher(args.matcher2, device='cuda', max_num_keypoints=args.num_kp2, 
    #                       ransac_reproj_thresh=args.ransac_reproj_thresh, ransac_iters=args.ransac_iters)  
     
    topk_dic = {}  
    with open(args.output_file, 'w') as result_writer: 
        
        for i in tqdm(range(len(test_fnames))): 
            
            query_im = cv2.imread(test_fnames[i], cv2.IMREAD_COLOR) 
               
            find_flag, result, match_im_name = compute_localization(args, query_im,indices[i], index_info_csv, matcher)
            
            # if not find_flag:
            #     args.apply_rot_aug = False
            #     find_flag, result, match_im_name = compute_localization(args, query_im, indices[i], index_info_csv, test_fnames[i], 
            #                                                         matcher2)
            

    ###########changed
    #         if find_flag:
    #              result_writer.write(os.path.basename(test_fnames[i]) + ' ' + os.path.basename(match_im_name) + \
    #                                 ' ' + str(result[0]) + ' ' + str(result[1]) + ' ' + str(result[2]) + \
    #                                 ' ' + str(result[3]) + '\n')
    #             #  base_map = cv2.imread(match_im_name)
    #             #  result = base_map[result[1]:result[3], result[0]:result[2]]
    #             #  cv2.imwrite('./result/result.jpg', result)
    #         else: 
    #             final_start_x = 0
    #             final_start_y = 0
    #             final_end_x = 5000
    #             final_end_y = 5000
    #             match_im_name = index_info_csv.iloc[[indices[i][0]]]['image_name'].values[0]
    #             result_writer.write(os.path.basename(test_fnames[i]) + ' ' + os.path.basename(match_im_name) + ' ' + \
    #                                                  str(final_start_x) + ' ' + str(final_start_y) + ' ' + str(final_end_x) +\
    #                                                      ' ' + str(final_end_y) + '\n')
                
    #         topk_dic[os.path.basename(test_fnames[i])] = [index_info_csv.iloc[[ind]]['image_name'].values[0] for ind in indices[i]]
        
    
    # if args.gt_file is not None:
    #     with open(args.output_file[:-4]+'_topk.txt', 'w') as result_writer: 
    #         for q in topk_dic:
    #             result_writer.write(q + ' ' + ' '.join(topk_dic[q]) + '\n')
                
    #     m_iou, acc = file_IoU_calculate(args.output_file, args.gt_file, args.output_file[:-4]+'_topk.txt')
    #     print(f"mIoU:{m_iou}    Acc: {acc}")