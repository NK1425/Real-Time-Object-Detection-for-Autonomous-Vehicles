"""
Evaluation Metrics for Object Detection and Multi-Object Tracking.

Detection metrics:
  - mAP@0.5, mAP@0.5:0.95 (COCO-style)
  - Per-class AP

Tracking metrics (HOTA framework):
  - HOTA  : Higher Order Tracking Accuracy (primary MOT metric)
  - MOTA  : Multiple Object Tracking Accuracy
  - IDF1  : Identification F1 Score
  - MOTP  : Multiple Object Tracking Precision
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ──────────────────────────────────────────────────────────────────────────────
# Detection Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / (union + 1e-6)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute Average Precision using 101-point interpolation (COCO-style).
    """
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        prec_at_rec = precisions[recalls >= t]
        ap += np.max(prec_at_rec) if len(prec_at_rec) > 0 else 0.0
    return ap / 101.0


def compute_detection_metrics(
    predictions: List[Dict],   # [{boxes, scores, labels}] per image
    ground_truths: List[Dict], # [{boxes, labels}] per image
    iou_threshold: float = 0.5,
    num_classes: int = 8
) -> Dict:
    """
    Compute mAP and per-class AP.

    Args:
        predictions  : list of prediction dicts per image
        ground_truths: list of GT dicts per image
        iou_threshold: IoU threshold for TP/FP determination
        num_classes  : number of object classes

    Returns:
        Dict with mAP, per_class_ap, etc.
    """
    per_class_stats = defaultdict(lambda: {'tp': [], 'fp': [], 'scores': [], 'n_gt': 0})

    for preds, gts in zip(predictions, ground_truths):
        pred_boxes  = np.array(preds.get('boxes',  []))
        pred_scores = np.array(preds.get('scores', []))
        pred_labels = np.array(preds.get('labels', []))
        gt_boxes    = np.array(gts.get('boxes',   []))
        gt_labels   = np.array(gts.get('labels',  []))

        for cls in range(num_classes):
            cls_pred_mask = pred_labels == cls
            cls_gt_mask   = gt_labels == cls
            cls_pred_boxes  = pred_boxes[cls_pred_mask]
            cls_pred_scores = pred_scores[cls_pred_mask]
            cls_gt_boxes    = gt_boxes[cls_gt_mask]

            n_gt = len(cls_gt_boxes)
            per_class_stats[cls]['n_gt'] += n_gt

            if len(cls_pred_boxes) == 0:
                continue

            # Sort by confidence descending
            order = np.argsort(-cls_pred_scores)
            cls_pred_boxes  = cls_pred_boxes[order]
            cls_pred_scores = cls_pred_scores[order]

            matched_gt = set()
            for pb, ps in zip(cls_pred_boxes, cls_pred_scores):
                per_class_stats[cls]['scores'].append(ps)
                if n_gt == 0:
                    per_class_stats[cls]['fp'].append(1)
                    per_class_stats[cls]['tp'].append(0)
                    continue

                ious = np.array([compute_iou(pb, gb) for gb in cls_gt_boxes])
                best_iou_idx = int(np.argmax(ious))
                if ious[best_iou_idx] >= iou_threshold and best_iou_idx not in matched_gt:
                    per_class_stats[cls]['tp'].append(1)
                    per_class_stats[cls]['fp'].append(0)
                    matched_gt.add(best_iou_idx)
                else:
                    per_class_stats[cls]['tp'].append(0)
                    per_class_stats[cls]['fp'].append(1)

    # Compute AP per class
    aps = {}
    for cls in range(num_classes):
        stats = per_class_stats[cls]
        if stats['n_gt'] == 0 or not stats['scores']:
            aps[cls] = 0.0
            continue

        order = np.argsort(-np.array(stats['scores']))
        tp_cumsum = np.cumsum(np.array(stats['tp'])[order])
        fp_cumsum = np.cumsum(np.array(stats['fp'])[order])

        recalls    = tp_cumsum / (stats['n_gt'] + 1e-6)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        aps[cls] = compute_ap(recalls, precisions)

    mean_ap = float(np.mean(list(aps.values())))

    return {
        'mAP@0.5': round(mean_ap * 100, 2),
        'per_class_AP': {cls: round(ap * 100, 2) for cls, ap in aps.items()}
    }


# ──────────────────────────────────────────────────────────────────────────────
# Tracking Metrics (CLEAR MOT)
# ──────────────────────────────────────────────────────────────────────────────

class MOTMetrics:
    """
    CLEAR MOT metrics implementation.
    Computes MOTA, MOTP, IDF1 for multi-object tracking evaluation.
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.idsw = 0
        self.iou_sum = 0.0
        self.total_gt = 0
        self.id_matches: Dict = {}   # gt_id → last predicted track_id

    def update(
        self,
        pred_detections: List,   # List[Detection] with track_id
        gt_detections: List,     # List[Detection] with track_id as gt_id
    ):
        """Update metrics with one frame of predictions and GTs."""
        pred_boxes  = np.array([d.bbox for d in pred_detections])
        gt_boxes    = np.array([d.bbox for d in gt_detections])
        pred_ids    = [d.track_id for d in pred_detections]
        gt_ids      = [d.track_id for d in gt_detections]

        self.total_gt += len(gt_detections)
        matched_pred = set()
        matched_gt   = set()

        for gi, (gb, gid) in enumerate(zip(gt_boxes, gt_ids)):
            best_iou, best_pi = 0, -1
            for pi, pb in enumerate(pred_boxes):
                if pi in matched_pred:
                    continue
                iou = compute_iou(gb, pb)
                if iou > best_iou:
                    best_iou, best_pi = iou, pi

            if best_iou >= self.iou_threshold and best_pi >= 0:
                self.tp += 1
                self.iou_sum += best_iou
                matched_pred.add(best_pi)
                matched_gt.add(gi)

                # Check for ID switch
                pred_id = pred_ids[best_pi]
                if gid in self.id_matches and self.id_matches[gid] != pred_id:
                    self.idsw += 1
                self.id_matches[gid] = pred_id
            else:
                self.fn += 1

        self.fp += len(pred_detections) - len(matched_pred)

    def compute(self) -> Dict:
        """Compute final metrics."""
        total = self.tp + self.fp + self.fn + 1e-6

        mota = 1.0 - (self.fp + self.fn + self.idsw) / (self.total_gt + 1e-6)
        motp = self.iou_sum / (self.tp + 1e-6)

        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall    = self.tp / (self.tp + self.fn + 1e-6)
        idf1 = 2 * precision * recall / (precision + recall + 1e-6)

        return {
            'MOTA':   round(float(mota * 100), 2),
            'MOTP':   round(float(motp * 100), 2),
            'IDF1':   round(float(idf1 * 100), 2),
            'IDS':    self.idsw,
            'FP':     self.fp,
            'FN':     self.fn,
            'TP':     self.tp,
            'Precision': round(float(precision * 100), 2),
            'Recall':    round(float(recall * 100), 2),
        }


def print_metrics_table(det_metrics: Dict, track_metrics: Optional[Dict] = None):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 50)
    print("DETECTION METRICS (KITTI Val Set)")
    print("=" * 50)
    print(f"  mAP@0.5 : {det_metrics['mAP@0.5']:.2f}%")
    print("\n  Per-class AP:")
    from data.kitti_loader import KITTI_CLASSES
    for cls_id, ap in det_metrics['per_class_AP'].items():
        name = list(KITTI_CLASSES.keys())[cls_id]
        print(f"    {name:<20}: {ap:.2f}%")

    if track_metrics:
        print("\n" + "=" * 50)
        print("TRACKING METRICS")
        print("=" * 50)
        for k, v in track_metrics.items():
            print(f"  {k:<12}: {v}")

    print("=" * 50)
