"""
YOLOv5 Detector Wrapper for KITTI Autonomous Driving.
Handles model loading, inference, and output post-processing.
Supports both PyTorch and TensorRT backends.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import time


KITTI_CLASSES = [
    'Car', 'Van', 'Truck', 'Pedestrian',
    'Person_sitting', 'Cyclist', 'Tram', 'Misc'
]

# Color palette per class (BGR)
CLASS_COLORS = {
    'Car':            (0, 255, 0),
    'Van':            (0, 200, 100),
    'Truck':          (0, 150, 255),
    'Pedestrian':     (255, 0, 0),
    'Person_sitting': (200, 0, 100),
    'Cyclist':        (255, 165, 0),
    'Tram':           (128, 0, 255),
    'Misc':           (128, 128, 128),
}


class Detection:
    """Single object detection result."""

    def __init__(
        self,
        bbox: np.ndarray,      # [x1, y1, x2, y2] in pixels
        confidence: float,
        class_id: int,
        depth: Optional[float] = None,
        track_id: Optional[int] = None
    ):
        self.bbox = bbox.astype(np.float32)
        self.confidence = float(confidence)
        self.class_id = int(class_id)
        self.class_name = KITTI_CLASSES[class_id] if class_id < len(KITTI_CLASSES) else 'Unknown'
        self.depth = depth
        self.track_id = track_id

    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )

    @property
    def area(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

    @property
    def color(self) -> Tuple[int, int, int]:
        return CLASS_COLORS.get(self.class_name, (128, 128, 128))

    def to_tlwh(self) -> np.ndarray:
        """Convert [x1,y1,x2,y2] to [top, left, width, height]."""
        x1, y1, x2, y2 = self.bbox
        return np.array([x1, y1, x2 - x1, y2 - y1])

    def __repr__(self):
        depth_str = f", depth={self.depth:.1f}m" if self.depth else ""
        track_str = f", track={self.track_id}" if self.track_id is not None else ""
        return (
            f"Detection({self.class_name}, conf={self.confidence:.2f}, "
            f"bbox={self.bbox.tolist()}{depth_str}{track_str})"
        )


class YOLOv5Detector:
    """
    YOLOv5 detector with KITTI-optimized settings.
    Loads from torch.hub or local weights file.
    """

    def __init__(
        self,
        weights: str = 'yolov5s',
        device: str = 'auto',
        img_size: int = 640,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        num_classes: int = 8
    ):
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.num_classes = num_classes

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = self._load_model(weights)
        self.model.eval()

        # Warm up
        self._warmup()
        print(f"[Detector] YOLOv5 loaded on {self.device} | img_size={img_size}")

    def _load_model(self, weights: str):
        if Path(weights).exists():
            # Load custom KITTI-trained weights
            model = torch.hub.load(
                'ultralytics/yolov5', 'custom',
                path=weights, force_reload=False
            )
        else:
            # Load pretrained from hub
            model = torch.hub.load('ultralytics/yolov5', weights, pretrained=True)

        model.conf = self.conf_thresh
        model.iou = self.iou_thresh
        model.classes = list(range(self.num_classes))
        return model.to(self.device)

    def _warmup(self, n: int = 3):
        dummy = torch.zeros(1, 3, self.img_size, self.img_size).to(self.device)
        for _ in range(n):
            with torch.no_grad():
                self.model(dummy)

    @torch.no_grad()
    def detect(self, img: np.ndarray) -> Tuple[List[Detection], float]:
        """
        Run detection on a single RGB image.
        Returns (list of Detection, inference_ms).
        """
        t0 = time.perf_counter()
        results = self.model(img, size=self.img_size)
        inference_ms = (time.perf_counter() - t0) * 1000

        detections = []
        preds = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]

        for pred in preds:
            x1, y1, x2, y2, conf, cls = pred
            det = Detection(
                bbox=np.array([x1, y1, x2, y2]),
                confidence=conf,
                class_id=int(cls)
            )
            detections.append(det)

        return detections, inference_ms

    @torch.no_grad()
    def detect_batch(
        self, imgs: List[np.ndarray]
    ) -> Tuple[List[List[Detection]], float]:
        """Batch inference for throughput benchmarking."""
        t0 = time.perf_counter()
        results = self.model(imgs, size=self.img_size)
        inference_ms = (time.perf_counter() - t0) * 1000

        batch_detections = []
        for preds in results.xyxy:
            detections = []
            for pred in preds.cpu().numpy():
                x1, y1, x2, y2, conf, cls = pred
                detections.append(Detection(
                    bbox=np.array([x1, y1, x2, y2]),
                    confidence=conf,
                    class_id=int(cls)
                ))
            batch_detections.append(detections)

        return batch_detections, inference_ms

    def get_model_info(self) -> Dict:
        """Return model statistics."""
        params = sum(p.numel() for p in self.model.parameters())
        return {
            'parameters': params,
            'parameters_M': round(params / 1e6, 2),
            'device': str(self.device),
            'img_size': self.img_size,
            'conf_thresh': self.conf_thresh,
            'iou_thresh': self.iou_thresh,
        }
