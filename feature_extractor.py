import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np
import argparse
import os
import glob

from vpr_model import VPRModel 
# Dataloader
from dataloaders.TestDataset import TestImageCropDataset
 
 
def input_transform(image_size=None):
    MEAN=[0.485, 0.456, 0.406]; STD=[0.229, 0.224, 0.225]
    if image_size:
        return T.Compose([
            T.Resize(image_size,  interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
 
"""
从给定的模型和dataloader中提取图像的全局特征描述符（descriptors）
"""
def get_descriptors(model, data_loader, device):
    all_descriptors = [] # 用于存储每个批次的描述符
    with torch.no_grad(): # 禁用梯度计算，避免占用内存和加速推理
        for inputs in data_loader: 
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_descriptors.append(outputs.cpu()) # 将输出的描述符添加到列表中
    return torch.cat(all_descriptors, dim=0)

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
    loaded_state_dict = torch.load(ckpt_path, map_location=torch.device('cuda:0'))
    if "state_dict" in loaded_state_dict.keys():
        loaded_state_dict = loaded_state_dict["state_dict"]
    model.load_state_dict(loaded_state_dict)
    model = model.eval()
    model = model.to('cuda')
    print(f"Loaded model from {ckpt_path} Successfully!")
    return model

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract DB image features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Model parameters
    parser.add_argument("--ckpt_path", type=str, default='./dino_salad.ckpt', help="Path to the checkpoint")
    parser.add_argument("--input_dir", type=str, required=True, default=None, help="Path to the image directory")
    parser.add_argument('--save_dir', type=str, required=True, default=None, help="Path to the save the extracted image features")
    parser.add_argument("--query", action='store_true', help="whether extract descriptors for query image")
    parser.add_argument('--image_size', nargs='+', type=lambda s: tuple(map(int, s.split(','))), required=True, help='Image size (int,)') 
    parser.add_argument("--crop_size", nargs='+', type=lambda s: tuple(map(int, s.split(','))), required=True, 
                        help="裁剪尺寸，格式: '高度,宽度', 例如 '256,256' ") # --crop_size 560 980
    parser.add_argument("--step_size", nargs='+', type=lambda s: tuple(map(int, s.split(','))), help="Step size for cropping the db images") # --crop_size 560 980
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers') 

    args = parser.parse_args()
 
    return args
  
if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True

    args = parse_args()
    
    model = load_model(args.ckpt_path) 
    model.eval() 
    device = torch.device('cuda')
    
    if not os.path.exists(args.save_dir):  
        os.makedirs(args.save_dir)
  
    im_paths = sorted(glob.glob(os.path.join(args.input_dir, '*.jpg'))) 
       
    for im_file_path in tqdm(im_paths):
        descriptors_allsize = []
        for image_size_input, crop_size, step_size  in zip(args.image_size, args.crop_size, args.step_size): 
            transform = input_transform(image_size=image_size_input)
            dataset = TestImageCropDataset(im_file_path, crop_size=crop_size, step_size=step_size, input_transform=transform)
            data_loader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True)
              
            descriptors = get_descriptors(model, data_loader, device)
            #对描述符张量的切片操作，确保只取前 len(dataset) 个描述符，并将 PyTorch 张量转化为 NumPy 数组
            descriptors = descriptors[:len(dataset)].numpy() 
            descriptors_allsize.append(descriptors)
        # Save results
        save_name = f'{os.path.splitext(os.path.basename(im_file_path))[0]}.npy'
        np.save(os.path.join(args.save_dir, save_name), np.concatenate(descriptors_allsize, axis=0))
