"""
Full AV Perception Pipeline: Detect → Track → Estimate Depth → Visualize

Integrates:
  - YOLOv5Detector (PyTorch or TensorRT backend)
  - ByteTracker (multi-object tracking)
  - MonocularDepthEstimator (distance estimation)
  - Visualizer (annotated frame output)

Usage:
    from inference.pipeline import AVPerceptionPipeline

    pipeline = AVPerceptionPipeline(
        weights='weights/yolov5_kitti.pt',
        engine_path='weights/yolov5_kitti_fp16.engine',  # optional TRT
        use_tensorrt=True,
    )
    results = pipeline.process_frame(frame)
"""

import cv2
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.detector import YOLOv5Detector, Detection
from models.tracker import ByteTracker
from models.depth_estimator import MonocularDepthEstimator
from inference.visualizer import Visualizer


@dataclass
class FrameResult:
    """Output of a single frame through the pipeline."""
    frame_id: int
    detections: List[Detection]
    annotated_frame: np.ndarray
    timings: Dict[str, float]   # ms per stage

    @property
    def total_ms(self) -> float:
        return sum(self.timings.values())

    @property
    def fps(self) -> float:
        return 1000.0 / (self.total_ms + 1e-6)

    @property
    def num_objects(self) -> int:
        return len(self.detections)

    def __repr__(self):
        return (
            f"FrameResult(id={self.frame_id}, objects={self.num_objects}, "
            f"fps={self.fps:.1f}, total={self.total_ms:.1f}ms, "
            f"timings={self.timings})"
        )


class AVPerceptionPipeline:
    """
    End-to-end autonomous vehicle perception pipeline.

    Supports two backends:
      - PyTorch: portable, works on any GPU
      - TensorRT: optimized for Nvidia edge devices (Jetson, T4, A100)

    Pipeline stages and typical latencies (on Jetson AGX Orin):
      1. Preprocess  :  ~1ms
      2. Detect      :  ~10ms (FP16 TRT), ~22ms (FP32 PyTorch)
      3. Track       :  ~2ms
      4. Depth est.  :  ~0.5ms
      5. Visualize   :  ~3ms
      Total          : ~16ms → ~62 FPS
    """

    def __init__(
        self,
        weights: str = 'yolov5s',
        engine_path: Optional[str] = None,
        use_tensorrt: bool = False,
        device: str = 'auto',
        img_size: int = 640,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        calib: Optional[Dict] = None,
        show_depth: bool = True,
        show_tracks: bool = True,
    ):
        self.img_size = img_size
        self.show_depth = show_depth
        self.show_tracks = show_tracks
        self.frame_id = 0
        self.fps_history: List[float] = []

        # --- Detector ---
        if use_tensorrt and engine_path and Path(engine_path).exists():
            from inference.trt_infer import TRTDetector
            self.detector = TRTDetector(
                engine_path=engine_path,
                img_size=img_size,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh
            )
            print(f"[Pipeline] Using TensorRT backend: {engine_path}")
        else:
            self.detector = YOLOv5Detector(
                weights=weights,
                device=device,
                img_size=img_size,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh
            )
            print(f"[Pipeline] Using PyTorch backend: {weights}")

        # --- Tracker ---
        self.tracker = ByteTracker(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=0.8
        )

        # --- Depth Estimator ---
        cam = calib or {}
        self.depth_estimator = MonocularDepthEstimator(
            fx=cam.get('fx', 721.54),
            fy=cam.get('fy', 721.54),
            cx=cam.get('cx', 609.56),
            cy=cam.get('cy', 172.85)
        )

        # --- Visualizer ---
        self.visualizer = Visualizer(show_depth=show_depth, show_tracks=show_tracks)

        print("[Pipeline] Ready.")

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Process one RGB frame through the full pipeline.

        Args:
            frame: HxWx3 RGB numpy array

        Returns:
            FrameResult with annotated frame and all detections
        """
        self.frame_id += 1
        timings = {}
        h, w = frame.shape[:2]

        # 1. Detect
        t0 = time.perf_counter()
        detections, det_ms = self.detector.detect(frame)
        timings['detect_ms'] = det_ms

        # 2. Track
        t1 = time.perf_counter()
        if detections:
            detections = self.tracker.update(detections)
        timings['track_ms'] = (time.perf_counter() - t1) * 1000

        # 3. Depth estimation
        t2 = time.perf_counter()
        if self.show_depth and detections:
            self.depth_estimator.estimate_batch(detections, img_h=h, img_w=w)
        timings['depth_ms'] = (time.perf_counter() - t2) * 1000

        # 4. Visualize
        t3 = time.perf_counter()
        annotated = self.visualizer.draw(frame.copy(), detections, self.frame_id)
        timings['viz_ms'] = (time.perf_counter() - t3) * 1000

        result = FrameResult(
            frame_id=self.frame_id,
            detections=detections,
            annotated_frame=annotated,
            timings=timings
        )

        # Track FPS rolling average
        self.fps_history.append(result.fps)
        if len(self.fps_history) > 100:
            self.fps_history.pop(0)

        return result

    def process_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        show_live: bool = False
    ) -> Dict:
        """
        Process a video file or camera stream.

        Args:
            input_path  : video file path or camera index (int as string)
            output_path : save annotated video if provided
            max_frames  : limit frames for benchmarking
            show_live   : display live window

        Returns:
            Stats dict with average FPS, mAP, etc.
        """
        try:
            source = int(input_path)  # webcam
        except ValueError:
            source = input_path       # file

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))

        all_fps, frame_count = [], 0
        print(f"[Pipeline] Processing: {input_path} ({total_frames} frames @ {fps_in:.0f}fps)")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.process_frame(frame_rgb)
                all_fps.append(result.fps)
                frame_count += 1

                annotated_bgr = cv2.cvtColor(result.annotated_frame, cv2.COLOR_RGB2BGR)

                if writer:
                    writer.write(annotated_bgr)

                if show_live:
                    cv2.imshow('AV Perception Pipeline', annotated_bgr)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if frame_count % 100 == 0:
                    avg_fps = np.mean(all_fps[-100:])
                    print(f"  Frame {frame_count}/{total_frames} | Avg FPS: {avg_fps:.1f} | "
                          f"Objects: {result.num_objects} | Timings: {result.timings}")

                if max_frames and frame_count >= max_frames:
                    break

        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

        avg_fps = float(np.mean(all_fps)) if all_fps else 0.0
        stats = {
            'frames_processed': frame_count,
            'avg_fps': round(avg_fps, 1),
            'min_fps': round(float(np.min(all_fps)), 1) if all_fps else 0,
            'max_fps': round(float(np.max(all_fps)), 1) if all_fps else 0,
            'output': output_path
        }
        print(f"\n[Pipeline] Done. Avg FPS: {avg_fps:.1f} over {frame_count} frames.")
        return stats

    @property
    def avg_fps(self) -> float:
        return float(np.mean(self.fps_history)) if self.fps_history else 0.0

    def reset(self):
        """Reset tracker state (e.g., new sequence)."""
        self.tracker.reset()
        self.frame_id = 0
        self.fps_history.clear()
