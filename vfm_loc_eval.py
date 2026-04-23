from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


VFM_LOC_ROOT = Path("/data/qq/Project/qq/VFM-Loc")
if str(VFM_LOC_ROOT) not in sys.path:
    sys.path.insert(0, str(VFM_LOC_ROOT))

from vfm_loc.engine import evaluate_retrieval  # noqa: E402
from vfm_loc.utils import canonical_query_labels  # noqa: E402
from vfm_loc.zero_shot import mean_by_id, pca_fit, pca_project, procrustes_align  # noqa: E402


class UniversityEvalDataset(Dataset):
    def __init__(self, folder: Path, transform, sample_ids: set[str] | None = None):
        self.transform = transform
        self.images = []
        self.labels = []
        self.sample_ids = []

        for subdir in sorted(path for path in folder.iterdir() if path.is_dir()):
            files = sorted(path for path in subdir.iterdir() if path.is_file())
            for file_path in files:
                self.images.append(file_path)
                self.sample_ids.append(subdir.name)
                label = -1 if sample_ids is not None and subdir.name not in sample_ids else int(subdir.name)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        with Image.open(self.images[index]) as image:
            image = image.convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(self.labels[index], dtype=torch.long)

    def get_sample_ids(self) -> set[str]:
        return set(self.sample_ids)


class University1652VfmLocEvaluator:
    def __init__(
        self,
        model,
        data_root: str,
        image_size: Sequence[int] = (336, 336),
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        batch_size: int = 64,
        num_workers: int = 4,
        pca_dim: int = 256,
        ranks: Iterable[int] = (1, 5, 10),
        step_size: int = 256,
        use_procrustes: bool = True,
        query_apply_adapter: bool = True,
        use_pca: bool = True,
    ):
        self.model = model
        self.data_root = Path(data_root)
        self.image_size = tuple(image_size)
        self.mean = list(mean)
        self.std = list(std)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pca_dim = pca_dim
        self.ranks = list(ranks)
        self.step_size = step_size
        self.use_procrustes = use_procrustes
        self.query_apply_adapter = query_apply_adapter
        self.use_pca = use_pca

    def _build_transform(self):
        return T.Compose(
            [
                T.Resize(self.image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def _build_loaders(self):
        transform = self._build_transform()
        query_root = self.data_root / "test" / "query_drone"
        gallery_root = self.data_root / "test" / "gallery_satellite"

        query_tmp = UniversityEvalDataset(query_root, transform=None)
        gallery_tmp = UniversityEvalDataset(gallery_root, transform=None)
        common_ids = query_tmp.get_sample_ids().intersection(gallery_tmp.get_sample_ids())
        if not common_ids:
            raise RuntimeError("No overlapping IDs found in University-1652 test split.")

        query_ds = UniversityEvalDataset(query_root, transform=transform, sample_ids=common_ids)
        gallery_ds = UniversityEvalDataset(gallery_root, transform=transform, sample_ids=common_ids)

        query_loader = DataLoader(
            query_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        gallery_loader = DataLoader(
            gallery_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return query_loader, gallery_loader

    @torch.no_grad()
    def _extract(self, loader, apply_adapter: bool):
        device = next(self.model.parameters()).device
        mixed_precision = device.type == "cuda"
        features, labels = [], []

        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=mixed_precision):
                feats = self.model(images, apply_adapter=apply_adapter)
            features.append(feats.to(torch.float32))
            labels.append(batch_labels.to(device))

        return torch.cat(features, dim=0), torch.cat(labels, dim=0)

    @torch.no_grad()
    def evaluate(self):
        query_loader, gallery_loader = self._build_loaders()
        query_features, query_labels = self._extract(query_loader, apply_adapter=self.query_apply_adapter)
        gallery_features, gallery_labels = self._extract(gallery_loader, apply_adapter=False)

        if self.use_pca:
            pca_dim = min(self.pca_dim, query_features.shape[-1], gallery_features.shape[-1])
            q_mean, q_proj = pca_fit(query_features)
            g_mean, g_proj = pca_fit(gallery_features)
            query_vectors = pca_project(query_features, q_mean, q_proj, pca_dim)
            gallery_vectors = pca_project(gallery_features, g_mean, g_proj, pca_dim)
        else:
            query_vectors = query_features.to(torch.float32)
            gallery_vectors = gallery_features.to(torch.float32)

        if self.use_procrustes:
            query_ids = canonical_query_labels(query_labels)
            q_unique, q_anchors = mean_by_id(query_vectors, query_ids)
            g_unique, g_anchors = mean_by_id(gallery_vectors, gallery_labels)
            common = sorted(set(q_unique.cpu().tolist()).intersection(set(g_unique.cpu().tolist())))
            if len(common) >= 2:
                q_map = {int(label.item()): idx for idx, label in enumerate(q_unique.cpu())}
                g_map = {int(label.item()): idx for idx, label in enumerate(g_unique.cpu())}
                q_pairs = torch.stack([q_anchors[q_map[item]] for item in common], dim=0)
                g_pairs = torch.stack([g_anchors[g_map[item]] for item in common], dim=0)
                rotation = procrustes_align(q_pairs, g_pairs)
                query_vectors = query_vectors @ rotation

        query_vectors = F.normalize(query_vectors, dim=-1)
        gallery_vectors = F.normalize(gallery_vectors, dim=-1)
        return evaluate_retrieval(
            query_features=query_vectors,
            reference_features=gallery_vectors,
            query_labels=query_labels,
            reference_labels=gallery_labels,
            ranks=self.ranks,
            step_size=self.step_size,
        )
