import os, json, random
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms as T
import pytorch_lightning as pl


IMAGENET_MEAN_STD = {
    'mean': [0.485, 0.456, 0.406],
    'std':  [0.229, 0.224, 0.225]
}

class GTAUAVDataModule(pl.LightningDataModule):
    def __init__(self,
                 json_path: str,
                 root_dir:   str,
                 batch_size: int = 32,
                 num_workers:int = 8,
                 num_augs:   int = 5,
                 image_size=(322,322)):
        super().__init__()
        self.json_path   = json_path
        self.root_dir    = root_dir
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.num_augs    = num_augs
        self.image_size  = image_size

    def setup(self, stage=None):
        # 这里只做 train
        self.train_ds = GTAUAVPairDataset(
            json_path   = self.json_path,
            root_dir    = self.root_dir,
            num_augs    = self.num_augs,
            image_size  = self.image_size
        )

    def train_dataloader(self):
        sampler = UniqueDroneBatchSampler(self.train_ds, batch_size=self.batch_size)
        return DataLoader(
            self.train_ds,
            batch_sampler   = sampler,
            num_workers     = self.num_workers,
            pin_memory      = True
        )


class GTAUAVPairDataset(Dataset):
    """
    把 GTAUAV 的 json 每一条 “drone 对应多个 sat” 
    flatten 成多条 (drone, sat) pair，每条当作一个 place。
    """
    def __init__(self,
                 json_path: str,
                 root_dir: str,
                 num_augs: int = 3,
                 image_size=(322,322),
                 mean_std=IMAGENET_MEAN_STD):
        super().__init__()
        self.root = root_dir
        self.num_augs = num_augs
        # transform for base(query=drone) and sat
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean_std['mean'], std=mean_std['std'])
        ])
        self.aug_transform = T.Compose([
            T.RandomRotation(360),
            T.RandAugment(num_ops=3),
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean_std['mean'], std=mean_std['std'])
        ])
        # 读 json 并 flatten
        with open(json_path, 'r') as f:
            records = json.load(f)
        self.pairs = []    # list of dict {drone_id, drone_path, sat_path}
        for rec in records:
            dname = rec['drone_img_name']
            drone_id = os.path.splitext(dname)[0]
            dpath = os.path.join(self.root, rec['drone_img_dir'], dname)
            # 正式 pos list + semipos list
            sats = rec.get('pair_pos_sate_img_list', []) \
                 + rec.get('pair_pos_semipos_sate_img_list', [])
            for s in sats:
                spath = os.path.join(self.root, rec['sate_img_dir'], s)
                self.pairs.append({
                    'drone_id': drone_id,
                    'drone_path': dpath,
                    'sat_path': spath
                })
        if not self.pairs:
            raise RuntimeError("没有任何 (drone, sat) pair")

        # 建立从 drone_id 到 indices 列表的映射，供 sampler 使用
        self.group2indices = {}
        for idx, p in enumerate(self.pairs):
            did = p['drone_id']
            self.group2indices.setdefault(did, []).append(idx)
        self.drone_ids = list(self.group2indices.keys())

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rec = self.pairs[idx]
        # load sat base
        try:
            sat = Image.open(rec['sat_path']).convert('RGB')
        except UnidentifiedImageError:
            sat = Image.new('RGB', (self.transform.transforms[0].size))
        base = self.transform(sat)
        # load drone query
        try:
            dr = Image.open(rec['drone_path']).convert('RGB')
        except UnidentifiedImageError:
            dr = Image.new('RGB', (self.transform.transforms[0].size))
        query = self.transform(dr)

        # 如果需要增强支路，继续在 sat 上做 num_augs 次 aug
        extras = [ self.aug_transform(sat) for _ in range(self.num_augs) ]

        patches = [base, query] + extras
        patches = torch.stack(patches, dim=0)   # shape = (1 + 1 + num_augs, C, H, W)
        # label: 给每一个 pair 一个全局唯一的 id
        label = torch.full((patches.size(0),),
                           fill_value=idx,
                           dtype=torch.long)
        return patches, label


class UniqueDroneBatchSampler(Sampler):
    """
    每个 epoch 遍历所有 (drone, sat) pair；同一 batch 内，保证不同 drone_id。
    """
    def __init__(self, dataset: GTAUAVPairDataset, batch_size: int):
        self.group2indices = dataset.group2indices      # drone_id -> [idx, ...]
        self.batch_size    = batch_size
        # 计算本 epoch 总样本数
        self.total_samples = sum(len(idxs) for idxs in self.group2indices.values())
        print(f"Total samples in GTAUAVPairDataset: {self.total_samples}")

    def __iter__(self):
        # 对每个 drone_id，拷贝并洗牌它的索引列表
        rem = {
            gid: idxs.copy()
            for gid, idxs in self.group2indices.items()
        }
        for idxs in rem.values():
            random.shuffle(idxs)

        # 初始时，所有 drone_id 都在“活跃”列表中
        active = [gid for gid, idxs in rem.items() if idxs]

        # 只要还有活跃的 drone，就继续产 batch
        while active:
            # 本 batch 要抽的 drone 数量
            B = min(self.batch_size, len(active))
            # 随机选 B 个不同 drone_id
            batch_gids = random.sample(active, B)

            batch = []
            for gid in batch_gids:
                idx = rem[gid].pop()    # 弹出一个 pair
                batch.append(idx)
                if not rem[gid]:        # 如果这个 drone 的 pair 全用完了
                    active.remove(gid)

            yield batch

    def __len__(self):
        # epoch 里能产出多少 batch
        return (self.total_samples + self.batch_size - 1) // self.batch_size