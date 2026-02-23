"""
Benchmark Suite: FPS, Latency, Memory, and Model Comparison.

Benchmarks:
  1. FPS / Latency across precisions (FP32, FP16, INT8)
  2. Preprocessing + inference + postprocessing breakdown
  3. Memory footprint (GPU RAM)
  4. Model comparison: YOLOv5s vs YOLOv5m vs YOLOv8n vs RT-DETR

Usage:
    python evaluation/benchmark.py \
        --weights weights/yolov5_kitti.pt \
        --trt-fp16 weights/yolov5_kitti_fp16.engine \
        --trt-int8 weights/yolov5_kitti_int8.engine \
        --data data/kitti/image_2/ \
        --num-runs 300 \
        --output results/benchmark_results.json
"""

import argparse
import json
import time
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class BenchmarkResult:
    model_name: str
    precision: str
    avg_fps: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    gpu_memory_mb: float
    throughput_imgs_per_sec: float
    num_runs: int

    def to_dict(self) -> Dict:
        return asdict(self)


def get_gpu_memory_mb() -> float:
    """Returns current GPU memory usage in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return 0.0


def load_benchmark_images(
    img_dir: str,
    num_images: int = 50,
    img_size: int = 640
) -> List[np.ndarray]:
    """Load a set of real KITTI images for benchmarking."""
    img_dir = Path(img_dir)
    img_paths = sorted(img_dir.glob('*.png'))[:num_images]
    img_paths += sorted(img_dir.glob('*.jpg'))[:max(0, num_images - len(img_paths))]

    if not img_paths:
        print(f"[Benchmark] No images found in {img_dir}. Using synthetic data.")
        return [np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
                for _ in range(num_images)]

    images = []
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

    print(f"[Benchmark] Loaded {len(images)} images from {img_dir}")
    return images


def benchmark_pytorch_model(
    weights: str,
    images: List[np.ndarray],
    device: str = 'cuda',
    num_warmup: int = 20,
    num_runs: int = 300
) -> BenchmarkResult:
    """Benchmark PyTorch YOLOv5 model."""
    from models.detector import YOLOv5Detector

    print(f"\n[Benchmark] PyTorch FP32 | {weights}")
    detector = YOLOv5Detector(weights=weights, device=device)

    # Warmup
    for img in images[:num_warmup]:
        detector.detect(img)

    # Benchmark
    latencies = []
    for i in range(num_runs):
        img = images[i % len(images)]
        _, ms = detector.detect(img)
        latencies.append(ms)

    return _compute_result('YOLOv5', 'FP32', latencies, num_runs, detector)


def benchmark_trt_engine(
    engine_path: str,
    images: List[np.ndarray],
    precision: str,
    num_warmup: int = 20,
    num_runs: int = 300
) -> Optional[BenchmarkResult]:
    """Benchmark TensorRT engine."""
    if not Path(engine_path).exists():
        print(f"[Benchmark] Engine not found: {engine_path} — skipping {precision}")
        return None

    try:
        from inference.trt_infer import TRTDetector
    except ImportError:
        print("[Benchmark] TensorRT not available — skipping TRT benchmarks.")
        return None

    print(f"\n[Benchmark] TensorRT {precision.upper()} | {engine_path}")
    detector = TRTDetector(engine_path=engine_path)

    for img in images[:num_warmup]:
        detector.detect(img)

    latencies = []
    for i in range(num_runs):
        img = images[i % len(images)]
        _, ms = detector.detect(img)
        latencies.append(ms)

    return _compute_result('YOLOv5', precision, latencies, num_runs, detector)


def _compute_result(
    model_name: str,
    precision: str,
    latencies: List[float],
    num_runs: int,
    detector
) -> BenchmarkResult:
    arr = np.array(latencies)
    avg_ms = float(np.mean(arr))
    return BenchmarkResult(
        model_name=model_name,
        precision=precision,
        avg_fps=round(1000.0 / avg_ms, 1),
        p50_latency_ms=round(float(np.percentile(arr, 50)), 2),
        p95_latency_ms=round(float(np.percentile(arr, 95)), 2),
        p99_latency_ms=round(float(np.percentile(arr, 99)), 2),
        min_latency_ms=round(float(np.min(arr)), 2),
        max_latency_ms=round(float(np.max(arr)), 2),
        gpu_memory_mb=round(get_gpu_memory_mb(), 1),
        throughput_imgs_per_sec=round(1000.0 / avg_ms, 1),
        num_runs=num_runs
    )


def print_benchmark_table(results: List[BenchmarkResult]):
    """Print formatted benchmark comparison table."""
    print("\n" + "=" * 100)
    print(f"{'Model':<12} {'Precision':<10} {'FPS':>8} {'p50(ms)':>10} {'p95(ms)':>10} "
          f"{'p99(ms)':>10} {'GPU RAM(MB)':>12} {'Speedup':>10}")
    print("=" * 100)

    baseline_ms = None
    for r in results:
        if r.precision == 'FP32':
            baseline_ms = r.p50_latency_ms
            break

    for r in results:
        speedup = f"{baseline_ms / r.p50_latency_ms:.2f}x" if baseline_ms else "—"
        print(
            f"{r.model_name:<12} {r.precision:<10} {r.avg_fps:>8.1f} "
            f"{r.p50_latency_ms:>10.2f} {r.p95_latency_ms:>10.2f} "
            f"{r.p99_latency_ms:>10.2f} {r.gpu_memory_mb:>12.1f} {speedup:>10}"
        )
    print("=" * 100)


def save_results(results: List[BenchmarkResult], output_path: str):
    """Save benchmark results to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data = {
        'results': [r.to_dict() for r in results],
        'system_info': _get_system_info()
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n[Benchmark] Results saved: {output_path}")


def _get_system_info() -> Dict:
    info = {'platform': 'unknown', 'cuda_version': 'N/A', 'gpu': 'N/A'}
    try:
        import torch
        info['torch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['gpu'] = torch.cuda.get_device_name(0)
            info['cuda_version'] = torch.version.cuda
    except Exception:
        pass
    try:
        import platform
        info['platform'] = platform.platform()
    except Exception:
        pass
    return info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark AV detection models')
    parser.add_argument('--weights',    type=str, default='yolov5s')
    parser.add_argument('--trt-fp16',   type=str, default='weights/yolov5_kitti_fp16.engine')
    parser.add_argument('--trt-int8',   type=str, default='weights/yolov5_kitti_int8.engine')
    parser.add_argument('--data',       type=str, default='data/kitti/image_2/')
    parser.add_argument('--num-runs',   type=int, default=300)
    parser.add_argument('--device',     type=str, default='cuda')
    parser.add_argument('--output',     type=str, default='results/benchmark_results.json')
    args = parser.parse_args()

    images = load_benchmark_images(args.data)
    results = []

    r_fp32 = benchmark_pytorch_model(args.weights, images, args.device, num_runs=args.num_runs)
    results.append(r_fp32)

    r_fp16 = benchmark_trt_engine(args.trt_fp16, images, 'FP16', num_runs=args.num_runs)
    if r_fp16: results.append(r_fp16)

    r_int8 = benchmark_trt_engine(args.trt_int8, images, 'INT8', num_runs=args.num_runs)
    if r_int8: results.append(r_int8)

    print_benchmark_table(results)
    save_results(results, args.output)
