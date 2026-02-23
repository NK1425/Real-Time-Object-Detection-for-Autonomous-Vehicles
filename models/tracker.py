"""
ByteTrack Multi-Object Tracker Integration.
ByteTrack associates detections across frames using IoU + Kalman Filter.
Achieves state-of-the-art MOTA/HOTA on MOT benchmarks.

Reference: ByteTrack - Multi-Object Tracking by Associating Every Detection Box
           Zhang et al., ECCV 2022
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


@dataclass
class TrackState:
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


@dataclass
class KalmanBoxTracker:
    """
    Kalman Filter based bounding box tracker.
    State: [cx, cy, s, r, vx, vy, vs]
      cx, cy = center coords
      s = scale (area)
      r = aspect ratio (fixed)
      vx, vy, vs = velocities
    """

    count: int = 0

    def __init__(self, bbox: np.ndarray, class_id: int, confidence: float):
        from numpy.linalg import matrix_rank

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.class_id = class_id
        self.confidence = confidence
        self.state = TrackState.NEW
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.time_since_update = 0
        self.history: List[np.ndarray] = []

        # Kalman filter matrices (7-state model)
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0

        self.H = np.zeros((4, 7))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        self.R = np.diag([1., 1., 10., 10.])
        self.Q = np.diag([1., 1., 1., 1., 0.01, 0.01, 0.0001])
        self.P = np.diag([10., 10., 10., 10., 10000., 10000., 10000.])

        self.x = self._bbox_to_z(bbox)
        self.x = np.concatenate([self.x, np.zeros((3, 1))], axis=0)

    def _bbox_to_z(self, bbox: np.ndarray) -> np.ndarray:
        """[x1,y1,x2,y2] → [cx, cy, s, r]"""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2, y1 + h / 2
        s = w * h
        r = w / (h + 1e-6)
        return np.array([[cx], [cy], [s], [r]])

    def _z_to_bbox(self, x: np.ndarray) -> np.ndarray:
        """[cx, cy, s, r] → [x1,y1,x2,y2]"""
        cx, cy, s, r = x[0, 0], x[1, 0], x[2, 0], x[3, 0]
        w = np.sqrt(max(s * r, 1e-6))
        h = s / (w + 1e-6)
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

    def predict(self) -> np.ndarray:
        """Kalman predict step."""
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        self.history.append(self._z_to_bbox(self.x))
        return self.history[-1]

    def update(self, bbox: np.ndarray, confidence: float):
        """Kalman update step with new measurement."""
        self.time_since_update = 0
        self.history.clear()
        self.hits += 1
        self.hit_streak += 1
        self.confidence = confidence

        z = self._bbox_to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def get_state(self) -> np.ndarray:
        return self._z_to_bbox(self.x)


def iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between two sets of bboxes."""
    if len(bboxes_a) == 0 or len(bboxes_b) == 0:
        return np.zeros((len(bboxes_a), len(bboxes_b)))

    ax1, ay1, ax2, ay2 = bboxes_a[:, 0], bboxes_a[:, 1], bboxes_a[:, 2], bboxes_a[:, 3]
    bx1, by1, bx2, by2 = bboxes_b[:, 0], bboxes_b[:, 1], bboxes_b[:, 2], bboxes_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a[:, None] + area_b[None, :] - inter_area

    return inter_area / (union_area + 1e-6)


class ByteTracker:
    """
    ByteTrack multi-object tracker.

    Key insight: uses BOTH high-confidence AND low-confidence detections.
    Low-confidence detections are used as a second association stage,
    recovering objects in occlusion or with motion blur.

    Config:
        track_thresh  : confidence above which detection is 'high'
        track_buffer  : max frames to keep a lost track
        match_thresh  : IoU threshold for association
        min_box_area  : filter tiny detections (noise)
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        min_box_area: float = 10.0,
        frame_rate: int = 30
    ):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)

        self.tracked_tracks: List[KalmanBoxTracker] = []
        self.lost_tracks: List[KalmanBoxTracker] = []
        self.removed_tracks: List[KalmanBoxTracker] = []
        self.frame_id = 0

        KalmanBoxTracker.count = 0

    def update(
        self,
        detections: List,   # List[Detection] from detector.py
    ) -> List:
        """
        Update tracker with new detections.
        Returns detections with track_id assigned.
        """
        self.frame_id += 1

        # Split detections into high/low confidence
        high_dets = [d for d in detections
                     if d.confidence >= self.track_thresh and d.area >= self.min_box_area]
        low_dets  = [d for d in detections
                     if d.confidence < self.track_thresh and d.area >= self.min_box_area]

        # Predict existing tracks
        for track in self.tracked_tracks + self.lost_tracks:
            track.predict()

        # ---- Stage 1: Match high-confidence dets to tracked tracks ----
        matched, unmatched_tracks, unmatched_dets = self._associate(
            self.tracked_tracks, high_dets, self.match_thresh
        )

        for t_idx, d_idx in matched:
            det = high_dets[d_idx]
            self.tracked_tracks[t_idx].update(det.bbox, det.confidence)
            self.tracked_tracks[t_idx].state = TrackState.TRACKED

        # ---- Stage 2: Match low-confidence dets to unmatched tracks ----
        unmatched_track_objs = [self.tracked_tracks[i] for i in unmatched_tracks]
        matched2, still_unmatched, _ = self._associate(
            unmatched_track_objs, low_dets, 0.5
        )

        for t_idx, d_idx in matched2:
            det = low_dets[d_idx]
            unmatched_track_objs[t_idx].update(det.bbox, det.confidence)
            unmatched_track_objs[t_idx].state = TrackState.TRACKED

        # ---- Stage 3: Try to re-associate lost tracks ----
        lost_matched, _, unmatched_new = self._associate(
            self.lost_tracks, [high_dets[i] for i in unmatched_dets], self.match_thresh
        )
        for t_idx, d_idx in lost_matched:
            det = high_dets[unmatched_dets[d_idx]]
            self.lost_tracks[t_idx].update(det.bbox, det.confidence)
            self.lost_tracks[t_idx].state = TrackState.TRACKED
            self.tracked_tracks.append(self.lost_tracks[t_idx])

        # ---- Init new tracks for unmatched high-confidence dets ----
        for d_idx in unmatched_new:
            det = high_dets[unmatched_dets[d_idx]]
            new_track = KalmanBoxTracker(det.bbox, det.class_id, det.confidence)
            new_track.state = TrackState.NEW
            self.tracked_tracks.append(new_track)

        # ---- Update lost/removed ----
        newly_lost = [self.tracked_tracks[i] for i in still_unmatched
                      if i < len(self.tracked_tracks)]
        for t in newly_lost:
            t.state = TrackState.LOST

        self.lost_tracks = [
            t for t in self.lost_tracks
            if t.state != TrackState.TRACKED and t.time_since_update <= self.max_time_lost
        ] + newly_lost

        self.tracked_tracks = [
            t for t in self.tracked_tracks
            if t.state == TrackState.TRACKED or t.state == TrackState.NEW
        ]

        # Assign track IDs back to detections
        self._assign_ids_to_detections(detections)

        return [d for d in detections if d.track_id is not None]

    def _associate(
        self,
        tracks: List[KalmanBoxTracker],
        detections: List,
        thresh: float
    ) -> Tuple[List, List, List]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        track_bboxes = np.array([t.get_state() for t in tracks])
        det_bboxes = np.array([d.bbox for d in detections])

        iou_matrix = iou_batch(track_bboxes, det_bboxes)
        cost_matrix = 1.0 - iou_matrix

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched, unmatched_tracks, unmatched_dets = [], [], []

        matched_set_t, matched_set_d = set(), set()
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= thresh:
                matched.append((r, c))
                matched_set_t.add(r)
                matched_set_d.add(c)

        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_set_t]
        unmatched_dets   = [i for i in range(len(detections)) if i not in matched_set_d]

        return matched, unmatched_tracks, unmatched_dets

    def _assign_ids_to_detections(self, detections: List):
        """Match active track states back to detection objects by bbox overlap."""
        for det in detections:
            best_iou = 0.3
            best_track = None
            for track in self.tracked_tracks:
                state_bbox = track.get_state()
                iou = iou_batch(
                    det.bbox[None], state_bbox[None]
                )[0, 0]
                if iou > best_iou:
                    best_iou = iou
                    best_track = track
            if best_track is not None:
                det.track_id = best_track.id

    def get_active_track_count(self) -> int:
        return len(self.tracked_tracks)

    def reset(self):
        self.tracked_tracks.clear()
        self.lost_tracks.clear()
        self.removed_tracks.clear()
        self.frame_id = 0
        KalmanBoxTracker.count = 0
