---
title: Real-Time Object Detection for Autonomous Vehicles
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
pinned: true
license: mit
short_description: YOLOv5 + ByteTrack + Depth | 78 mAP | TensorRT
---

# Real-Time Object Detection for Autonomous Vehicles

**YOLOv5 · ByteTrack · Monocular Depth · TensorRT · KITTI**

- 78 mAP@0.5 on KITTI | 45 FPS (FP32) → 134 FPS (INT8 TensorRT)
- ByteTrack multi-object tracking with persistent IDs
- Monocular depth estimation per object (±15% error)
- Deployed on Jetson AGX Orin for edge inference

[GitHub →](https://github.com/NK1425/Real-Time-Object-Detection-for-Autonomous-Vehicles)
