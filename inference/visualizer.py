"""
Visualization module for AV perception pipeline.
Draws bounding boxes, track IDs, class labels, depth warnings,
and an optional Bird's Eye View (BEV) overlay.
"""

import cv2
import numpy as np
from typing import List, Optional, Dict
from models.detector import Detection, CLASS_COLORS
from models.depth_estimator import WARNING_COLORS


class Visualizer:
    """
    Draws annotated overlays on detection frames.

    Features:
      - Color-coded bounding boxes per class
      - Track ID labels (ByteTrack)
      - Distance / depth annotation
      - Safety warning overlays (CRITICAL / WARNING / CAUTION)
      - FPS counter + stats panel
      - Optional Bird's Eye View (BEV)
    """

    def __init__(
        self,
        show_depth: bool = True,
        show_tracks: bool = True,
        show_bev: bool = False,
        font_scale: float = 0.55,
        line_thickness: int = 2
    ):
        self.show_depth = show_depth
        self.show_tracks = show_tracks
        self.show_bev = show_bev
        self.font_scale = font_scale
        self.thickness = line_thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        frame_id: int = 0,
        fps: Optional[float] = None,
        extra_info: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Draw all annotations on frame.

        Args:
            frame      : RGB image (modified in-place)
            detections : list of Detection objects with optional track_id and depth
            frame_id   : current frame number
            fps        : current FPS to display
            extra_info : dict of additional text to display in stats panel

        Returns:
            Annotated RGB frame
        """
        for det in detections:
            self._draw_detection(frame, det)

        self._draw_stats_panel(frame, detections, frame_id, fps, extra_info)

        if self.show_bev and detections:
            frame = self._overlay_bev(frame, detections)

        return frame

    def _draw_detection(self, frame: np.ndarray, det: Detection):
        x1, y1, x2, y2 = det.bbox.astype(int)
        color = det.color

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)

        # Build label string
        parts = [det.class_name, f"{det.confidence:.2f}"]
        if self.show_tracks and det.track_id is not None:
            parts.append(f"#{det.track_id}")
        if self.show_depth and det.depth is not None:
            parts.append(f"{det.depth:.1f}m")
        label = " | ".join(parts)

        # Draw label background
        (text_w, text_h), baseline = cv2.getTextSize(
            label, self.font, self.font_scale, self.thickness
        )
        label_y1 = max(y1 - text_h - baseline - 4, 0)
        cv2.rectangle(
            frame,
            (x1, label_y1),
            (x1 + text_w + 4, y1),
            color, -1
        )

        # Draw label text (white on colored background)
        cv2.putText(
            frame, label,
            (x1 + 2, y1 - baseline - 2),
            self.font, self.font_scale,
            (255, 255, 255), self.thickness - 1,
            cv2.LINE_AA
        )

        # Draw depth warning indicator
        if det.depth is not None:
            from models.depth_estimator import MonocularDepthEstimator
            estimator = MonocularDepthEstimator.__new__(MonocularDepthEstimator)
            warning = estimator.get_distance_warning(det.depth)
            if warning:
                warn_color = WARNING_COLORS.get(warning, (255, 255, 255))
                cv2.rectangle(
                    frame,
                    (x1, y1), (x2, y2),
                    warn_color, self.thickness + 1
                )
                # Warning label top-right corner of bbox
                cv2.putText(
                    frame, warning,
                    (x2 - 80, y1 + 20),
                    self.font, 0.45,
                    warn_color, 1, cv2.LINE_AA
                )

        # Draw center dot
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(frame, (cx, cy), 3, color, -1)

    def _draw_stats_panel(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        frame_id: int,
        fps: Optional[float],
        extra_info: Optional[Dict]
    ):
        """Draw semi-transparent stats panel in top-left corner."""
        h, w = frame.shape[:2]

        # Count per class
        class_counts: Dict[str, int] = {}
        for det in detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

        lines = [f"Frame: {frame_id}"]
        if fps is not None:
            lines.append(f"FPS: {fps:.1f}")
        lines.append(f"Objects: {len(detections)}")
        for cls, cnt in sorted(class_counts.items()):
            lines.append(f"  {cls}: {cnt}")
        if extra_info:
            for k, v in extra_info.items():
                lines.append(f"{k}: {v}")

        panel_w = 200
        panel_h = 20 + len(lines) * 22
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (panel_w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        for i, line in enumerate(lines):
            cv2.putText(
                frame, line,
                (10, 22 + i * 22),
                self.font, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA
            )

    def _overlay_bev(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        bev_size: int = 200,
        max_depth: float = 60.0,
        lateral_range: float = 20.0
    ) -> np.ndarray:
        """
        Overlay a Bird's Eye View (BEV) minimap in the bottom-right corner.
        Objects are plotted by estimated depth (forward) and lateral position.
        """
        h, w = frame.shape[:2]
        bev = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)

        # Draw grid lines
        for d_line in range(0, bev_size, bev_size // 6):
            cv2.line(bev, (0, d_line), (bev_size, d_line), (30, 30, 30), 1)
        for l_line in range(0, bev_size, bev_size // 4):
            cv2.line(bev, (l_line, 0), (l_line, bev_size), (30, 30, 30), 1)

        # Ego vehicle marker (bottom center)
        ego_x, ego_y = bev_size // 2, bev_size - 10
        cv2.rectangle(bev, (ego_x - 5, ego_y - 8), (ego_x + 5, ego_y + 8), (0, 255, 0), -1)
        cv2.putText(bev, "EGO", (ego_x - 12, ego_y - 12),
                    self.font, 0.3, (0, 255, 0), 1)

        # Plot detected objects
        for det in detections:
            if det.depth is None:
                continue

            # Lateral position estimate from bbox center
            cx_norm = (det.bbox[0] + det.bbox[2]) / 2 / w - 0.5  # -0.5 to 0.5
            lateral_m = cx_norm * lateral_range * 2

            # BEV pixel coords
            bev_x = int(ego_x + (lateral_m / lateral_range) * (bev_size // 2))
            bev_y = int(ego_y - (det.depth / max_depth) * (bev_size - 20))
            bev_x = np.clip(bev_x, 0, bev_size - 1)
            bev_y = np.clip(bev_y, 0, bev_size - 1)

            color_bgr = det.color
            cv2.circle(bev, (bev_x, bev_y), 4, color_bgr, -1)
            if det.track_id is not None:
                cv2.putText(bev, str(det.track_id),
                            (bev_x + 5, bev_y - 5),
                            self.font, 0.25, color_bgr, 1)

        # Draw border
        cv2.rectangle(bev, (0, 0), (bev_size - 1, bev_size - 1), (100, 100, 100), 1)
        cv2.putText(bev, "BEV", (4, 12), self.font, 0.35, (200, 200, 200), 1)

        # Paste BEV onto frame bottom-right
        bx = w - bev_size - 10
        by = h - bev_size - 10
        frame[by:by + bev_size, bx:bx + bev_size] = bev

        return frame
