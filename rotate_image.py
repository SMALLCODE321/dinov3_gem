import cv2
import os
import numpy as np
import random
from glob import glob

def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, M, (width, height))
    return rotated_image

# 指定包含图像的文件夹路径
folder_path = '/home/lyf/data/km2-online/test_image_little_crop'

# 获取文件夹内所有的jpg图像路径
image_paths = glob(os.path.join(folder_path, '*.jpg'))

# 对每张图像执行旋转操作
for image_path in image_paths:
    # 读取图像
    img = cv2.imread(image_path)
    
    # 设定旋转的角度（例如90度）
    angle = random.randint(0, 180)
    
    # 进行旋转
    rotated_img = rotate_image(img, angle)
    
    # 输出到新的文件（这里直接覆盖原文件）
    output_path = os.path.join('/home/lyf/data/km2-online/test_image_little_rotate_noextend', os.path.basename(image_path))
    cv2.imwrite(output_path, rotated_img)

print("所有图像旋转完成。")