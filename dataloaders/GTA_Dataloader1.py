import os, json, random
from collections import defaultdict
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms as T
import pytorch_lightning as pl

IMAGENET_MEAN_STD = {
    'mean': [0.485, 0.456, 0.406],
    'std':  [0.229, 0.224, 0.225]
}

class GTAUAVPlaceDataModule(pl.LightningDataModule):
    def __init__(self,
                 json_path: str,
                 root_dir:   str,
                 batch_size: int = 8,
                 num_workers:int = 4,
                 num_augs:   int = 5,
                 image_size=(322,322),
                 max_drones_per_place: int = 10):
        super().__init__()
        self.json_path    = json_path
        self.root_dir     = root_dir
        self.batch_size   = batch_size
        self.num_workers  = num_workers
        self.num_augs     = num_augs
        self.image_size   = image_size
        self.max_drones   = max_drones_per_place

    def setup(self, stage=None):
        self.train_ds = GTAUAVPlaceDataset(
            json_path             = self.json_path,
            root_dir              = self.root_dir,
            num_augs              = self.num_augs,
            image_size            = self.image_size,
            max_drones_per_place  = self.max_drones
        )

    def train_dataloader(self):
        sampler = UniqueSatBatchSampler(self.train_ds, batch_size=self.batch_size)
        return DataLoader(
            self.train_ds,
            batch_sampler = sampler,
            num_workers   = self.num_workers,
            pin_memory    = True
        )

class GTAUAVPlaceDataset(Dataset):
    def __init__(self,
                 json_path: str,
                 root_dir:   str,
                 num_augs:   int = 3,
                 image_size=(322,322),
                 max_drones_per_place: int = 10,
                 mean_std=IMAGENET_MEAN_STD):
        super().__init__()
        self.root     = root_dir
        self.num_augs = num_augs
        self.max_drones = max_drones_per_place

        # transforms
        self.base_tf = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean_std['mean'], std=mean_std['std'])
        ])
        self.aug_tf  = T.Compose([
            T.RandomRotation(360),
            T.RandAugment(num_ops=3),
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean_std['mean'], std=mean_std['std'])
        ])

        # 扫一遍 json，构建 sat->drone 映射
        sat2path   = {}
        drone2path = {}
        sat2drones = defaultdict(set)

        with open(json_path, 'r') as f:
            recs = json.load(f)

        for rec in recs:
            dname = rec['drone_img_name']
            did   = os.path.splitext(dname)[0]
            drone2path[did] = os.path.join(root_dir, rec['drone_img_dir'], dname)

            sats = rec.get('pair_pos_sate_img_list', []) + rec.get('pair_pos_semipos_sate_img_list', [])
            for s in sats:
                sid = os.path.splitext(s)[0]
                sat2path[sid]      = os.path.join(root_dir, rec['sate_img_dir'], s)
                sat2drones[sid].add(did)

        self.sat_ids    = sorted(sat2path.keys())
        self.sat2label = {sid: i for i, sid in enumerate(self.sat_ids)}
        self.sat2idx   = {sid: i for i, sid in enumerate(self.sat_ids)}
        self.sat2path   = sat2path
        self.drone2path = drone2path
        self.sat2drones = {sid: list(sat2drones[sid]) for sid in self.sat_ids}

    def __len__(self):
        return len(self.sat_ids)

    def __getitem__(self, index):
        sid      = self.sat_ids[index]
        place_id = self.sat2label[sid]

        # 1) 载入并转换 sat_base + sat_extras
        try:
            sat = Image.open(self.sat2path[sid]).convert('RGB')
        except UnidentifiedImageError:
            sat = Image.new('RGB', (self.base_tf.transforms[0].size))
        sat_base   = self.base_tf(sat)
        sat_extras = [self.aug_tf(sat) for _ in range(self.num_augs)]

        # 2) 载入对应的 drone 图，若超出上限，随机采样
        drone_ids = self.sat2drones[sid]
        if len(drone_ids) > self.max_drones:
            drone_ids = random.sample(drone_ids, self.max_drones)

        drone_imgs = []
        for did in drone_ids:
            try:
                dr = Image.open(self.drone2path[did]).convert('RGB')
            except UnidentifiedImageError:
                dr = Image.new('RGB', (self.base_tf.transforms[0].size))
            drone_imgs.append(self.base_tf(dr))

        # 3) 如果 drone 数不足，用卫星图的 aug 填充
        n_missing = self.max_drones - len(drone_imgs)
        for _ in range(n_missing):
            drone_imgs.append(self.aug_tf(sat))  # 用 sat 做增强当“伪 drone”

        # 4) 拼成一个固定长度的 Tensor: 1 + num_augs + max_drones
        patches = torch.stack([sat_base, *sat_extras, *drone_imgs], dim=0)
        labels  = torch.full((patches.size(0),), place_id, dtype=torch.long)
        return patches, labels


class UniqueSatBatchSampler(Sampler):
    """
    在一个 epoch 中遍历所有 sat place，
    组 batch 时尽量保证同一个 batch 内不同 place 之间不共用同一 drone_id，
    如果在填满 batch_size 前再也找不到新的“不冲突” place，就先产出当前 batch。
    """
    def __init__(self, dataset: GTAUAVPlaceDataset, batch_size: int):
        self.batch_size  = batch_size
        self.sat_ids     = dataset.sat_ids
        self.sat2idx     = dataset.sat2idx
        self.sat2drones  = dataset.sat2drones
        self.total_plcs  = len(self.sat_ids)
        print(f"Total places in GTAUAVPlaceDataset: {self.total_plcs}")

    def __len__(self):
        return (self.total_plcs + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        sat_list = self.sat_ids.copy()
        random.shuffle(sat_list)

        batches       = []
        cur_batch     = []
        cur_drone_set = set()

        for sid in sat_list:
            drones = set(self.sat2drones[sid])

            # 如果能加且没满，就加
            if len(cur_batch) < self.batch_size and drones.isdisjoint(cur_drone_set):
                cur_batch.append(sid)
                cur_drone_set |= drones
            else:
                # 把已有的 batch 收下
                if cur_batch:
                    batches.append(cur_batch)
                # 重开一个 batch
                cur_batch     = [sid]
                cur_drone_set = set(drones)

        # 最后一批
        if cur_batch:
            batches.append(cur_batch)

        # 将 sat_id 批次转为索引
        for batch in batches:
            yield [ self.sat2idx[sid] for sid in batch ]