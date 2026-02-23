"""
KITTI Dataset Loader
Parses KITTI object detection labels, velodyne point clouds, and camera calibration.
Supports 2D/3D bounding boxes and converts to YOLOv5 format.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml


KITTI_CLASSES = {
    'Car': 0, 'Van': 1, 'Truck': 2,
    'Pedestrian': 3, 'Person_sitting': 4,
    'Cyclist': 5, 'Tram': 6, 'Misc': 7
}

IGNORE_CLASSES = {'DontCare'}


class KITTICalibration:
    """Parse and store KITTI camera calibration matrices."""

    def __init__(self, calib_path: str):
        self.data = self._parse_calib(calib_path)
        self.P2 = self.data['P2'].reshape(3, 4)      # Left color camera projection
        self.R0_rect = self.data['R0_rect'].reshape(3, 3)
        self.Tr_velo_cam = self.data['Tr_velo_to_cam'].reshape(3, 4)

    def _parse_calib(self, path: str) -> Dict:
        data = {}
        with open(path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line or ':' not in line:
                    continue
                key, val = line.split(':', 1)
                data[key.strip()] = np.array(
                    [float(x) for x in val.strip().split()]
                )
        return data

    @property
    def fx(self): return self.P2[0, 0]

    @property
    def fy(self): return self.P2[1, 1]

    @property
    def cx(self): return self.P2[0, 2]

    @property
    def cy(self): return self.P2[1, 2]

    def project_3d_to_2d(self, pts_3d: np.ndarray) -> np.ndarray:
        """Project 3D camera coords to 2D image coords."""
        pts_3d_hom = np.hstack([pts_3d, np.ones((pts_3d.shape[0], 1))])
        pts_2d_hom = (self.P2 @ pts_3d_hom.T).T
        pts_2d = pts_2d_hom[:, :2] / pts_2d_hom[:, 2:3]
        return pts_2d


class KITTIObject:
    """Single KITTI object annotation."""

    def __init__(self, label_line: str):
        parts = label_line.strip().split()
        self.type = parts[0]
        self.truncated = float(parts[1])
        self.occluded = int(parts[2])
        self.alpha = float(parts[3])
        # 2D bounding box (left, top, right, bottom)
        self.bbox_2d = np.array([float(x) for x in parts[4:8]])
        # 3D dimensions (height, width, length)
        self.dimensions = np.array([float(x) for x in parts[8:11]])
        # 3D location in camera coords (x, y, z)
        self.location = np.array([float(x) for x in parts[11:14]])
        # Rotation around Y-axis
        self.rotation_y = float(parts[14])
        self.score = float(parts[15]) if len(parts) > 15 else -1.0

    @property
    def class_id(self) -> Optional[int]:
        return KITTI_CLASSES.get(self.type, None)

    @property
    def is_valid(self) -> bool:
        return (
            self.type not in IGNORE_CLASSES
            and self.class_id is not None
            and self.truncated <= 0.8
            and self.occluded <= 2
        )

    @property
    def depth(self) -> float:
        """Distance from camera in meters."""
        return float(self.location[2])

    def to_yolo(self, img_w: int, img_h: int) -> Optional[np.ndarray]:
        """Convert to YOLOv5 format: [class_id, cx, cy, w, h] normalized."""
        if not self.is_valid:
            return None
        x1, y1, x2, y2 = self.bbox_2d
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        if w <= 0 or h <= 0:
            return None
        return np.array([self.class_id, cx, cy, w, h], dtype=np.float32)


class KITTIDataset(Dataset):
    """
    PyTorch Dataset for KITTI object detection.
    Returns images and YOLOv5-format labels.
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        img_size: int = 640,
        augment: bool = True,
        config_path: str = 'configs/kitti_yolov5.yaml'
    ):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == 'train'

        self.img_dir = self.root / 'image_2'
        self.label_dir = self.root / 'label_2'
        self.calib_dir = self.root / 'calib'

        self.ids = self._load_split_ids(split)
        print(f"[KITTI] Loaded {len(self.ids)} samples for split='{split}'")

    def _load_split_ids(self, split: str) -> List[str]:
        split_file = self.root / f'ImageSets/{split}.txt'
        if split_file.exists():
            with open(split_file) as f:
                return [line.strip() for line in f if line.strip()]
        # Fallback: use all available images
        return [p.stem for p in sorted(self.img_dir.glob('*.png'))]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict:
        sample_id = self.ids[idx]
        img = self._load_image(sample_id)
        labels = self._load_labels(sample_id, img.shape[:2])
        calib = self._load_calib(sample_id)

        img, labels = self._preprocess(img, labels)

        return {
            'image': torch.from_numpy(img).permute(2, 0, 1).float() / 255.0,
            'labels': torch.from_numpy(labels) if len(labels) else torch.zeros((0, 5)),
            'sample_id': sample_id,
            'calib': {
                'fx': calib.fx, 'fy': calib.fy,
                'cx': calib.cx, 'cy': calib.cy
            }
        }

    def _load_image(self, sample_id: str) -> np.ndarray:
        path = self.img_dir / f'{sample_id}.png'
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_labels(self, sample_id: str, img_shape: Tuple) -> np.ndarray:
        path = self.label_dir / f'{sample_id}.txt'
        if not path.exists():
            return np.zeros((0, 5), dtype=np.float32)

        h, w = img_shape
        labels = []
        with open(path) as f:
            for line in f:
                obj = KITTIObject(line)
                yolo_label = obj.to_yolo(w, h)
                if yolo_label is not None:
                    labels.append(yolo_label)

        return np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)

    def _load_calib(self, sample_id: str) -> KITTICalibration:
        path = self.calib_dir / f'{sample_id}.txt'
        return KITTICalibration(str(path))

    def _preprocess(
        self, img: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        img = cv2.resize(img, (self.img_size, self.img_size))
        return img, labels


def build_dataloader(
    root: str,
    split: str = 'train',
    img_size: int = 640,
    batch_size: int = 16,
    num_workers: int = 4,
    augment: bool = True
) -> DataLoader:
    dataset = KITTIDataset(root, split, img_size, augment)

    def collate_fn(batch):
        images = torch.stack([b['image'] for b in batch])
        labels = [b['labels'] for b in batch]
        ids = [b['sample_id'] for b in batch]
        calibs = [b['calib'] for b in batch]
        return {'images': images, 'labels': labels, 'ids': ids, 'calibs': calibs}

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == 'train')
    )
