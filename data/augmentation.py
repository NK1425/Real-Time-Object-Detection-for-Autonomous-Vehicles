"""
Augmentation Pipeline for KITTI Autonomous Driving Dataset.
Includes standard augmentations + driving-specific ones:
  - Weather simulation (rain, fog, night)
  - Mosaic mixing
  - Photometric distortions
"""

import cv2
import numpy as np
import random
from typing import Tuple, List, Optional


class WeatherAugmentation:
    """Simulate adverse weather conditions for robustness training."""

    @staticmethod
    def add_rain(img: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Overlay rain streaks on image."""
        img = img.copy()
        h, w = img.shape[:2]
        num_drops = int(intensity * 800)
        for _ in range(num_drops):
            x1 = random.randint(0, w - 1)
            y1 = random.randint(0, h - 1)
            length = random.randint(10, 30)
            angle = random.uniform(-20, 20)
            x2 = int(x1 + length * np.sin(np.radians(angle)))
            y2 = int(y1 + length * np.cos(np.radians(angle)))
            x2, y2 = np.clip(x2, 0, w - 1), np.clip(y2, 0, h - 1)
            cv2.line(img, (x1, y1), (x2, y2), (200, 200, 220), 1)
        return img

    @staticmethod
    def add_fog(img: np.ndarray, intensity: float = 0.4) -> np.ndarray:
        """Add fog effect using weighted blend with white."""
        fog = np.full_like(img, 255)
        return cv2.addWeighted(img, 1 - intensity, fog, intensity, 0)

    @staticmethod
    def simulate_night(img: np.ndarray, gamma: float = 2.5) -> np.ndarray:
        """Darken image to simulate night driving."""
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ]).astype(np.uint8)
        return cv2.LUT(img, table)

    @staticmethod
    def add_sun_glare(img: np.ndarray) -> np.ndarray:
        """Simulate sun glare in upper region of image."""
        img = img.copy()
        h, w = img.shape[:2]
        cx = random.randint(w // 4, 3 * w // 4)
        cy = random.randint(0, h // 3)
        radius = random.randint(50, 150)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), radius, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (101, 101), 0)
        for c in range(3):
            img[:, :, c] = np.clip(
                img[:, :, c] + (mask * 180).astype(np.uint8), 0, 255
            )
        return img.astype(np.uint8)


class PhotometricDistortion:
    """Color and lighting distortions."""

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)

        # Random brightness
        if random.random() < 0.5:
            delta = random.uniform(-32, 32)
            img += delta

        # Random contrast
        if random.random() < 0.5:
            alpha = random.uniform(0.5, 1.5)
            img *= alpha

        img = np.clip(img, 0, 255).astype(np.uint8)

        # Convert to HSV for saturation/hue jitter
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)

        if random.random() < 0.5:
            img_hsv[:, :, 1] *= random.uniform(0.5, 1.5)

        if random.random() < 0.5:
            img_hsv[:, :, 0] += random.uniform(-18, 18)
            img_hsv[:, :, 0] %= 180

        img_hsv = np.clip(img_hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)


class MosaicAugmentation:
    """
    YOLOv5-style mosaic: combines 4 images into one.
    Significantly improves detection of small objects.
    """

    def __init__(self, img_size: int = 640):
        self.img_size = img_size

    def __call__(
        self,
        images: List[np.ndarray],
        labels_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert len(images) == 4, "Mosaic requires exactly 4 images"
        s = self.img_size
        cx = random.randint(s // 4, 3 * s // 4)
        cy = random.randint(s // 4, 3 * s // 4)

        mosaic_img = np.zeros((2 * s, 2 * s, 3), dtype=np.uint8)
        all_labels = []

        placements = [
            (0, 0, cx, cy),         # top-left
            (cx, 0, 2 * s, cy),     # top-right
            (0, cy, cx, 2 * s),     # bottom-left
            (cx, cy, 2 * s, 2 * s)  # bottom-right
        ]

        for i, (x1, y1, x2, y2) in enumerate(placements):
            img = cv2.resize(images[i], (x2 - x1, y2 - y1))
            mosaic_img[y1:y2, x1:x2] = img

            labels = labels_list[i].copy()
            if len(labels):
                pw, ph = x2 - x1, y2 - y1
                labels[:, 1] = (labels[:, 1] * pw + x1) / (2 * s)
                labels[:, 2] = (labels[:, 2] * ph + y1) / (2 * s)
                labels[:, 3] = labels[:, 3] * pw / (2 * s)
                labels[:, 4] = labels[:, 4] * ph / (2 * s)
                all_labels.append(labels)

        mosaic_img = cv2.resize(mosaic_img, (s, s))
        all_labels_arr = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0, 5))

        # Scale labels back after resize
        all_labels_arr[:, 1:] = np.clip(all_labels_arr[:, 1:], 0, 1) if len(all_labels_arr) else all_labels_arr

        return mosaic_img, all_labels_arr


class HorizontalFlip:
    """Horizontal flip with label adjustment."""

    def __call__(
        self, img: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        img = np.fliplr(img).copy()
        if len(labels):
            labels = labels.copy()
            labels[:, 1] = 1.0 - labels[:, 1]
        return img, labels


class KITTIAugmentPipeline:
    """
    Full augmentation pipeline for KITTI training.
    Applied in order: weather → photometric → geometric → mosaic.
    """

    def __init__(self, img_size: int = 640, mosaic_prob: float = 0.5):
        self.img_size = img_size
        self.mosaic_prob = mosaic_prob
        self.weather = WeatherAugmentation()
        self.photometric = PhotometricDistortion()
        self.mosaic = MosaicAugmentation(img_size)
        self.hflip = HorizontalFlip()

    def apply_single(
        self, img: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Weather augmentation (low probability)
        r = random.random()
        if r < 0.1:
            img = self.weather.add_rain(img)
        elif r < 0.2:
            img = self.weather.add_fog(img)
        elif r < 0.25:
            img = self.weather.simulate_night(img)

        # Photometric distortion
        img = self.photometric(img)

        # Horizontal flip
        if random.random() < 0.5:
            img, labels = self.hflip(img, labels)

        return img, labels

    def apply_mosaic(
        self,
        images: List[np.ndarray],
        labels_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.mosaic(images, labels_list)
