from pathlib import Path
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# NOTE: you need to download the mapillary_sls dataset from  https://github.com/FrederikWarburg/mapillary_sls
# make sure the path where the mapillary_sls validation dataset resides on your computer is correct.
# the folder named train_val should reside in DATASET_ROOT path (that's the only folder you need from mapillary_sls)
# I hardcoded the groundtruth for image to image evaluation, otherwise it would take ages to run the groundtruth script at each epoch.
# DATASET_ROOT = './datasets/tzb_val/images/'

# path_obj = Path(DATASET_ROOT)
# if not path_obj.exists():
#     raise Exception('Please make sure the path to tzb_val dataset is correct')

# if not path_obj.joinpath('train_val'):
#     raise Exception(f'Please make sure the directory train_val from tzb_val dataset is situated in the directory {DATASET_ROOT}')

class TZB(Dataset):
    def __init__(self, data_root='./datasets/tzb_val1', input_transform = None):
        
        self.data_root = data_root
        self.input_transform = input_transform
        
        # hard coded reference image names, this avoids the hassle of listing them at each epoch.
        self.dbImages = np.load(os.path.join(data_root, 'tzb_val_dbImages.npy'))
        
        # hard coded query image names.
        self.qImages = np.load(os.path.join(data_root, 'tzb_val_qImages.npy'))
        
        # hard coded index of query images
        self.qIdx = np.load(os.path.join(data_root, 'tzb_val_qIdx.npy'))
        
        # hard coded groundtruth (correspondence between each query and its matches)
        self.pIdx = np.load(os.path.join(data_root, 'tzb_val_pIdx.npy'), allow_pickle=True)
        
        # concatenate reference images then query images so that we can use only one dataloader
        self.images = np.concatenate((self.dbImages, self.qImages[self.qIdx]))
        
        # we need to keeo the number of references so that we can split references-queries 
        # when calculating recall@K
        self.num_references = len(self.dbImages)
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_root, 'images', self.images[index]))

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)