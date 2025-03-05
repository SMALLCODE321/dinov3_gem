from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset 
import torchvision.transforms as T
 
def resize_and_pad(image, min_size=322, max_size=980):
    # 获取图像的原始尺寸
    width, height = image.size
    max_dim = max(width, height)
    
    # 如果最长边在范围内，则保持原尺寸
    if min_size <= max_dim <= max_size:
        target_size = max_dim
    else:
        # 计算新的尺寸，使最长边在指定范围内
        if max_dim < min_size:
            target_size = min_size
        else:
            target_size = max_size
    
    # 计算缩放比例
    scale = target_size / max_dim
    new_width = int(width * scale)
    new_height = int(height * scale) 
    # 调整图像大小
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 计算填充后的正方形尺寸，使其能被14整除
    padded_size = ((max(new_width, new_height) + 13) // 14) * 14
    
    # 创建新图像并将调整后的图像粘贴到中心
    new_image = Image.new('RGB', (padded_size, padded_size), (0, 0, 0))
    top_left_x = (padded_size - new_width) // 2
    top_left_y = (padded_size - new_height) // 2
    new_image.paste(resized_image, (top_left_x, top_left_y))
    
    return new_image
  
def input_transform(image_size=None):
    MEAN=[0.485, 0.456, 0.406]; STD=[0.229, 0.224, 0.225]
    if image_size:
        return T.Compose([
            T.Resize((image_size,image_size),  interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
             
class TestListDataset(Dataset):
    def __init__(self, im_list=[], image_size=None): 
        self.input_transform = input_transform(image_size)
        self.images = im_list
        self.image_size = image_size
     
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB') 
       
        if self.image_size is None:
            img = resize_and_pad(img)
            
        if self.input_transform:
            img = self.input_transform(img) 
        return img

    def __len__(self):
        return len(self.images)
 
 
class TestImageCropDataset(Dataset):
    def __init__(self, im_path, crop_size=[560], step_size=None, input_transform = None):
        
        self.crop_size = crop_size
        if step_size is None:
            self.step_size = crop_size
        else:
            self.step_size = step_size
            
        self.input_transform = input_transform  
        self.im = self.image_loader(im_path)
        # generate the dataframe contraining images metadata
        self.patches_corr = self.__getdataframes() 
        
    def __generate_crop_corrs(self, image_size): 
        H, W = image_size 
        patches = []  
        for patch_size, step in zip(self.crop_size, self.step_size): 
            for top in range(0, H, step):  
                for left in range(0, W, step):  
                    bottom = min(top + patch_size, H)
                    right = min(left + patch_size, W) 
                    if bottom - top < patch_size:
                        top = max(bottom - patch_size, 0)
                    if right - left < patch_size:
                        left = max(right - patch_size, 0) 
                    patch = (left, top, right, bottom) 
                    patches.append(patch)    
        return patches      
    
    def __getdataframes(self):   
        width, height  = self.im.size
        patches_corr = self.__generate_crop_corrs((height, width)) 
         
        return patches_corr
          
    
    def __getitem__(self, index):
        corrs = self.patches_corr[index]    
        img = self.im.crop((corrs[0], corrs[1], corrs[2], corrs[3]) )
        
        if self.input_transform is not None:
            img = self.input_transform(img)
  
        return img
    
    def __len__(self):
        return len(self.patches_corr)
     
    @staticmethod
    def image_loader(path):
        try:
            return Image.open(path).convert('RGB')
        except UnidentifiedImageError:
            print(f'Image {path} could not be loaded')
            return Image.new('RGB', (224, 224))