import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F 
from third_party.AdaLAM.adalam import AdalamFilter
from PIL import Image
import kornia.feature as KF
from kornia_moons.feature import laf_from_opencv_SIFT_kpts

def sift_to_rootsift(x: torch.Tensor, eps=1e-6) -> torch.Tensor:
    x = torch.nn.functional.normalize(x, p=1, dim=-1, eps=eps)
    x.clip_(min=eps).sqrt_()
    return torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps)

    
class PointMatcher:
    
     # Matcher configurations
    flann_matcher_config = {
        'index_params': dict(algorithm=1, trees=5),  # Fixed typo here ('trees' instead of 'tree')
        'search_params': dict(checks=50),
        'knn_neighbors': 2
    }
    
    lightglue_matcher_config = {
        'params': dict(filter_threshold=0.2, width_confidence=-1,
                       depth_confidence=-1, mp=True)
    }
    
    adalam_matcher_config = {
        'params': dict(max_iters=1000, confidence=0.99, error=3.0)
    }
    
    matcher_config_dict = {
        'flann': flann_matcher_config,
        'lightglue': lightglue_matcher_config,
        'adalam': adalam_matcher_config,
        'dualsoftmax': {},
        'bf': {},
    }
    
    def __init__(self, feature_type='sift', matcher_type='flann', matcher_config=None, device='cuda'):
        self.feature_type = feature_type  
        self.matcher_type = matcher_type
        self.device = device 
            
        if matcher_config is None:
            self._init_matcher_config()
        else:
            self.matcher_config = matcher_config
            
        self.detector, self.matcher = self._create_feature_extractor_and_matcher()
        
    def _init_matcher_config(self):
        self.matcher_config = self.matcher_config_dict[self.matcher_type]
        
    def _create_feature_extractor_and_matcher(self):
        # Create feature extractor based on the type
        if self.feature_type == 'sift':
            if self.matcher_type == 'adalam':
                detector = cv2.SIFT_create(nfeatures=8000, contrastThreshold=1e-5)
            else:
                detector = cv2.SIFT_create() 
                
        elif self.feature_type == 'orb': 
                detector = cv2.ORB_create()
                  
        elif self.feature_type =='disk':
            detector = KF.DISK.from_pretrained("depth").to(self.device)
         
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")

        # Create matcher based on the type
        if self.matcher_type == 'flann': 
            # FLANN parameters for SIFT 
            index_params = self.matcher_config['index_params']
            search_params = self.matcher_config['search_params']
            matcher = cv2.FlannBasedMatcher(index_params, search_params)  
            
        elif self.matcher_type == 'lightglue':  
            matcher = KF.LightGlueMatcher(self.feature_type).eval().to(self.device) 
               
        elif self.matcher_type == 'adalam':   
            matcher = AdalamFilter()  
            
        elif self.matcher_type == 'bf':
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
             
        else: 
            raise ValueError(f"Unsupported matcher type: {self.matcher_type}")
            
        return detector, matcher
        
    def extract_features(self, img):
        """
        Extracts keypoints and descriptors from the image.

        Args:
            img (numpy.ndarray): The input image in grayscale.

        Returns:
            keypoints (list): The list of keypoints detected.
            descriptors (numpy.ndarray): The descriptors for the keypoints.
        """
        #创建空列表keypoints和descriptors用于存储提取的关键点和描述符
        keypoints, descriptors = [], []
        #如果特征类型是"SIFT"或"ORB"，使用OpenCV的detectAndCompute方法同时检测关键点并计算描述符。
        """
        如果特征类型是"DISK"，首先将图像从BGR转换为RGB格式。
        将图像转换为32位浮点格式并归一化到[0, 1]范围。
        添加批次维度并调整维度顺序以符合Kornia库的要求。
        使用Kornia的DISK检测器提取特征，获取关键点和描述符。  
        """
        if self.feature_type in ['sift', 'orb']: 
            keypoints, descriptors = self.detector.detectAndCompute(img, None)  
        
    
        elif self.feature_type == 'disk':
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            # Convert the image to 32-bit floating point format (RGB32 equivalent)
            img_rgb32 = img_rgb.astype(np.float32) / 255.0  # Normalize to [0, 1] if needed 
            # Add a batch dimension (shape: 1 x H x W x C)
            img_rgb32 = torch.tensor(img_rgb32[None, ...], device=self.device).permute(0,3,1,2) 
            features = self.detector(img_rgb32, 8192, pad_if_not_divisible=True)
          
            keypoints, descriptors = features[0].keypoints,features[0].descriptors
            del img_rgb32
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
                
        return keypoints, descriptors
    
    def match(self, points1, points2, *args, **kwargs):
        """
        Matches descriptors using the configured matcher.

        Args:
            descriptors1 (numpy.ndarray): Descriptors from the first image.
            descriptors2 (numpy.ndarray): Descriptors from the second image.

        Returns:
            matches (list): List of DMatch objects representing the matches.
        """ 
        #从输入的points1和points2中提取关键点和描述符
        keypoints1, descriptors1 = points1
        keypoints2, descriptors2 = points2
        
        #创建空列表src_pts和dst_pts用于存储匹配的源点和目标点
        matches, src_pts, dst_pts = [], [],[]  # 源图像中的点 
        #如果任一图像的关键点为空，直接返回空列表
        if len(keypoints1)==0 or len(keypoints2)==0:
            return src_pts, dst_pts 
        
        """
        1.使用knnMatch方法找到描述符的k近邻匹配。
        通过距离比率测试筛选良好匹配，并将匹配点添加到src_pts和dst_pts。

        2.bf匹配
        使用match方法找到描述符的匹配。
        将所有匹配点添加到src_pts和dst_pts

        3.使用Adalam匹配器
        提取关键点的位置、角度和大小信息。
        使用match_and_filter方法进行匹配和过滤。
        将匹配点添加到src_pts和dst_pts
        """
        if self.matcher_type == 'flann':
            # Use knnMatch for FLANN-based matcher
            matches = self.matcher.knnMatch(descriptors1, descriptors2, k=self.matcher_config['knn_neighbors'])   
            for match in matches:  
                m, n = match
                if m.distance < 0.75 * n.distance:
                    src_pts.append(keypoints1[m.queryIdx].pt)
                    dst_pts.append(keypoints2[m.trainIdx].pt) 
                    
        elif self.matcher_type == 'bf':
            matches = self.matcher.match(descriptors1, descriptors2)
            for match in matches:  
                src_pts.append(keypoints1[match.queryIdx].pt)
                dst_pts.append(keypoints2[match.trainIdx].pt)
                                
        elif self.matcher_type == 'adalam':
            size_1 = kwargs.get('size_1', None) 
            size_2 = kwargs.get('size_2', None) 
            pts1 = np.array([k.pt for k in keypoints1], dtype=np.float32)
            ors1 = np.array([k.angle for k in keypoints1], dtype=np.float32)
            scs1 = np.array([k.size for k in keypoints1], dtype=np.float32)
            
            pts2 = np.array([k.pt for k in keypoints2], dtype=np.float32)
            ors2 = np.array([k.angle for k in keypoints2], dtype=np.float32)
            scs2 = np.array([k.size for k in keypoints2], dtype=np.float32)
            
            matches = self.matcher.match_and_filter(k1=pts1, k2=pts2,
                                    o1=ors1, o2=ors2,
                                    d1=descriptors1, d2=descriptors2,
                                    s1=scs1, s2=scs2,
                                    im1shape=size_1, im2shape=size_2).cpu().numpy()

            src_pts, dst_pts = pts1[matches[:, 0]], pts2[matches[:, 1]]
              
        elif self.matcher_type == 'lightglue':
            hw1 = torch.tensor(kwargs.get('size_1', None), device=self.device)
            hw2 = torch.tensor(kwargs.get('size_2', None), device=self.device)
            if self.feature_type == 'sift':
                with torch.inference_mode():
                    lafs1 = laf_from_opencv_SIFT_kpts(keypoints1, self.device)
                    lafs2 = laf_from_opencv_SIFT_kpts(keypoints2, self.device)
                    descs1 = sift_to_rootsift(torch.from_numpy(descriptors1)).to(self.device)
                    descs2 = sift_to_rootsift(torch.from_numpy(descriptors2)).to(self.device)  
            
            elif self.feature_type == 'disk':
                lafs1 = KF.laf_from_center_scale_ori(keypoints1[None], torch.ones(1, len(keypoints1), 1, 1, device=self.device))
                lafs2 = KF.laf_from_center_scale_ori(keypoints2[None], torch.ones(1, len(keypoints2), 1, 1, device=self.device))
                descs1, descs2 = descriptors1, descriptors2
            
            else:
                raise ValueError(f"Unsupported feature type: {self.feature_type} for lightglue")  
              
            dists, idxs = self.matcher(descs1, descs2, lafs1, lafs2, hw1=hw1, hw2=hw2)
            
            del lafs1
            del lafs2
            del descs1
            del descs2
            
            keypoints1 = keypoints1.cpu().numpy()
            keypoints2 = keypoints2.cpu().numpy()
            for d in idxs:
                src_pts.append(keypoints1[d[0]])
                dst_pts.append(keypoints2[d[1]]) 
        else:
            raise ValueError(f"Unsupported matcher type: {self.matcher_type}")
          
        return src_pts, dst_pts 

    def match_images(self, img1, img2):
        """
        Calculate match points between two images.

        Args:
            img1 (numpy.ndarray): The first input image 
            img2 (numpy.ndarray): The second input image

        Returns:
            keypoints1 (list): Keypoints from the first image.
            keypoints2 (list): Keypoints from the second image.
            matches (list): List of DMatch objects representing the matches.
        """
        # Extract features from both images
        
        #对两幅图像分别调用extract_features方法提取关键点和描述符
        #如果使用的是"Adalam"或"LightGlue"匹配器，需要提供图像尺寸作为额外参数
        if self.feature_type in ['sift', 'orb', 'disk']:
            keypoints1, descriptors1 = self.extract_features(img1)
            keypoints2, descriptors2 = self.extract_features(img2)  
            # Match descriptors
            if self.matcher_type in ['adlam', 'lightglue']:
                src_pts, dst_pts = self.match((keypoints1, descriptors1), (keypoints2, descriptors2), 
                                              size_1=img1.shape[:2], size_2=img2.shape[:2])       
            else:
                src_pts, dst_pts = self.match((keypoints1, descriptors1), (keypoints2, descriptors2))
              
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
              
        return src_pts, dst_pts 
       