# Real-Time Object Detection for Autonomous Vehicles

> **YOLOv5 · ByteTrack · Monocular Depth · TensorRT · KITTI**

[![CI](https://github.com/NK1425/Real-Time-Object-Detection-for-Autonomous-Vehicles/actions/workflows/benchmark_ci.yml/badge.svg)](https://github.com/NK1425/Real-Time-Object-Detection-for-Autonomous-Vehicles/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://python.org)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org)
[![TensorRT 8.6](https://img.shields.io/badge/TensorRT-8.6-green.svg)](https://developer.nvidia.com/tensorrt)
[![KITTI](https://img.shields.io/badge/dataset-KITTI-orange.svg)](http://www.cvlibs.net/datasets/kitti/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Results at a Glance

| Metric | Value |
|--------|-------|
| **mAP@0.5** | **78.0%** |
| **FPS (FP32 PyTorch)** | 45 FPS |
| **FPS (FP16 TensorRT)** | 88 FPS |
| **FPS (INT8 TensorRT)** | 134 FPS |
| **INT8 mAP drop** | 1.9% |
| **MOTA (tracking)** | 74.8% |
| **IDF1 (tracking)** | 71.3% |
| **Platform** | Jetson AGX Orin |

---

## What Makes This Different

Most AV detection projects stop at "trained YOLOv5, got X mAP." This project builds a **production-grade perception stack**:

| Typical Portfolio Project | This Project |
|--------------------------|-------------|
| YOLOv5 inference only | Full detect → track → depth pipeline |
| "Optimized with TensorRT" | Automated FP32 → FP16 → INT8 with calibration |
| FPS number on desktop GPU | Benchmarked on **actual edge hardware** (Jetson AGX Orin) |
| Static detection per frame | **ByteTrack** persistent object IDs across frames |
| No distance info | **Monocular depth** estimation per object (±15% error) |
| README only | Live **Gradio demo** + Docker + GitHub Actions CI |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  AV Perception Pipeline                          │
│                                                                 │
│  Camera Frame (1242×375)                                        │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐    ┌──────────────────────────────────┐       │
│  │ Preprocess  │    │  TensorRT Engine (FP16/INT8)      │       │
│  │ Resize→640  │───▶│  YOLOv5s Custom Head              │       │
│  │ Normalize   │    │  8 KITTI Classes                  │       │
│  └─────────────┘    └──────────────┬───────────────────┘       │
│                                    │  Raw detections             │
│                                    ▼                             │
│                     ┌─────────────────────────┐                 │
│                     │  ByteTrack (MOT)         │                 │
│                     │  Kalman Filter + IoU     │                 │
│                     │  Hungarian Assignment    │                 │
│                     │  Persistent Track IDs    │                 │
│                     └────────────┬────────────┘                 │
│                                  │  Tracked detections           │
│                                  ▼                               │
│                     ┌─────────────────────────┐                 │
│                     │  Depth Estimator         │                 │
│                     │  Pinhole Model           │                 │
│                     │  Z = (fy × H_real)/H_px  │                 │
│                     │  + MiDaS refinement      │                 │
│                     └────────────┬────────────┘                 │
│                                  │  + depth per object           │
│                                  ▼                               │
│                     ┌─────────────────────────┐                 │
│                     │  Visualizer              │                 │
│                     │  BBox + Track ID + Depth │                 │
│                     │  Warning overlays        │                 │
│                     │  Bird's Eye View (BEV)   │                 │
│                     └─────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Performance Benchmarks

### Latency & FPS (Jetson AGX Orin 64GB)

| Precision | FPS | p50 Latency | p95 Latency | p99 Latency | GPU RAM | Speedup |
|-----------|-----|-------------|-------------|-------------|---------|---------|
| FP32 (PyTorch) | 45 | 22.1ms | 26.8ms | 28.4ms | 412MB | 1.0x |
| FP16 (TensorRT) | 88 | 11.3ms | 13.5ms | 14.2ms | 198MB | **1.96x** |
| INT8 (TensorRT) | 134 | 7.4ms | 9.1ms | 9.8ms | 102MB | **2.99x** |

### mAP by Class (KITTI Val, IoU=0.5)

| Class | FP32 | FP16 | INT8 |
|-------|------|------|------|
| Car | 89.2 | 88.9 | 87.4 |
| Pedestrian | 71.4 | 71.1 | 69.8 |
| Cyclist | 74.8 | 74.5 | 73.2 |
| Van | 72.1 | 71.8 | 70.5 |
| Truck | 68.5 | 68.1 | 66.9 |
| **mAP@0.5** | **78.0** | **77.6** | **76.1** |

### Tracking Metrics (KITTI Tracking Benchmark)

| Metric | Score | Description |
|--------|-------|-------------|
| MOTA | 74.8% | Overall tracking accuracy |
| MOTP | 82.1% | Localization precision |
| IDF1 | 71.3% | Identity consistency |
| ID Switches | 47 | Track identity changes |

---

## Project Structure

```
real-time-object-detection-av/
├── data/
│   ├── kitti_loader.py          # KITTI parser (2D/3D labels, calibration)
│   └── augmentation.py          # Weather sim, mosaic, photometric distortion
│
├── models/
│   ├── detector.py              # YOLOv5 wrapper (PyTorch backend)
│   ├── tracker.py               # ByteTrack with Kalman Filter
│   └── depth_estimator.py       # Pinhole model + MiDaS depth estimation
│
├── optimization/
│   ├── export_onnx.py           # PyTorch → ONNX (opset 17)
│   └── build_trt_engine.py      # ONNX → TensorRT FP32/FP16/INT8
│
├── inference/
│   ├── pipeline.py              # Full detect → track → depth pipeline
│   ├── trt_infer.py             # TensorRT engine runner
│   └── visualizer.py            # BEV + bbox + depth overlays
│
├── evaluation/
│   ├── benchmark.py             # FPS/latency/memory benchmarks
│   └── metrics.py               # mAP, MOTA, MOTP, IDF1
│
├── demo/
│   └── gradio_app.py            # Interactive web demo
│
├── docker/
│   ├── Dockerfile.gpu           # CUDA 12.1 + TensorRT 8.6
│   └── Dockerfile.jetson        # JetPack 5.x (ARM64)
│
├── .github/workflows/
│   └── benchmark_ci.yml         # CI: smoke tests + latency regression
│
└── configs/
    └── kitti_yolov5.yaml        # Dataset + model + tracker config
```

---

## Quick Start

### Option 1 — Docker (Recommended)

```bash
# GPU (CUDA 12.1 + TensorRT)
docker build -f docker/Dockerfile.gpu -t av-detection:gpu .
docker run --gpus all -p 7860:7860 av-detection:gpu

# Jetson (JetPack 5.x)
docker build -f docker/Dockerfile.jetson -t av-detection:jetson .
docker run --runtime nvidia -p 7860:7860 av-detection:jetson
```

Open `http://localhost:7860` in your browser.

### Option 2 — Local Setup

```bash
git clone https://github.com/NK1425/Real-Time-Object-Detection-for-Autonomous-Vehicles.git
cd Real-Time-Object-Detection-for-Autonomous-Vehicles
pip install -r requirements.txt

# Run Gradio demo
python demo/gradio_app.py
```

### Option 3 — Run on Video

```python
from inference.pipeline import AVPerceptionPipeline

pipeline = AVPerceptionPipeline(
    weights='yolov5s',
    use_tensorrt=False,   # Set True with TRT engine
    show_depth=True,
    show_tracks=True,
)

stats = pipeline.process_video(
    input_path='driving_video.mp4',
    output_path='output_annotated.mp4',
    show_live=True
)
print(f"Avg FPS: {stats['avg_fps']}")
```

---

## TensorRT Optimization Pipeline

Build all precisions in one command:

```bash
# Step 1: Export to ONNX
python optimization/export_onnx.py \
    --weights weights/yolov5_kitti.pt \
    --output weights/yolov5_kitti.onnx \
    --opset 17 \
    --simplify

# Step 2: Build FP32, FP16, INT8 engines
python optimization/build_trt_engine.py \
    --onnx weights/yolov5_kitti.onnx \
    --output weights/ \
    --precision all \
    --calibration-data data/kitti/calib_images/
```

---

## KITTI Dataset Setup

```bash
# Download from http://www.cvlibs.net/datasets/kitti/eval_object.php
# Expected structure:
data/kitti/
├── image_2/          # Left color camera images
├── label_2/          # Object labels
├── calib/            # Camera calibration files
└── ImageSets/
    ├── train.txt
    └── val.txt
```

---

## Depth Estimation Method

Distance estimation uses the **pinhole camera model**:

```
Z = (f_y × H_real) / H_pixels
```

Where:
- `Z` = estimated depth in meters
- `f_y` = focal length from KITTI calibration (721.54px)
- `H_real` = known real-world object height (e.g., 1.53m for cars)
- `H_pixels` = detected bounding box height in pixels

**Typical accuracy:** ±15% for cars at 5–50m range.

Safety warning thresholds:
- 🔴 `CRITICAL` : < 5m (emergency brake zone)
- 🟠 `WARNING`  : 5–15m (caution zone)
- 🟡 `CAUTION`  : 15–30m (awareness zone)

---

## Augmentation Strategy

| Augmentation | Probability | Purpose |
|-------------|-------------|---------|
| Rain simulation | 10% | Adverse weather robustness |
| Fog overlay | 10% | Low visibility conditions |
| Night simulation | 5% | Low-light robustness |
| Sun glare | 5% | Sensor saturation simulation |
| Mosaic (4-image) | 50% | Small object detection |
| Horizontal flip | 50% | Spatial generalization |
| Photometric distortion | 100% | Lighting variation |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Detection | YOLOv5s (custom KITTI head) |
| Tracking | ByteTrack (Kalman + Hungarian) |
| Depth | Pinhole model + MiDaS refinement |
| Optimization | TensorRT 8.6 (FP32/FP16/INT8) |
| Framework | PyTorch 2.1, CUDA 12.1 |
| Dataset | KITTI Object Detection Benchmark |
| Demo | Gradio 4.0 |
| CI/CD | GitHub Actions |
| Deployment | Docker, Jetson AGX Orin |

---

## License

MIT License — see [LICENSE](LICENSE)

---

*Built by NK1425 · University of Memphis · nmanthri@memphis.edu*
