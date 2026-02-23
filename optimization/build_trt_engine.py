"""
TensorRT Engine Builder: ONNX → TensorRT (FP32 / FP16 / INT8)

Full optimization pipeline:
  1. Parse ONNX model
  2. Build TensorRT engine with selected precision
  3. Serialize engine to .engine file
  4. Benchmark latency and throughput

Supports:
  - FP32: Full precision baseline
  - FP16: 2x speedup, ~0.5% mAP drop
  - INT8: 4x speedup with calibration, ~1-2% mAP drop

Usage:
    python optimization/build_trt_engine.py \
        --onnx weights/yolov5_kitti.onnx \
        --output weights/ \
        --precision all \
        --workspace 4 \
        --calibration-data data/kitti/calib_images/
"""

import os
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Optional


def build_engine(
    onnx_path: str,
    output_path: str,
    precision: str = 'fp16',
    workspace_gb: int = 4,
    max_batch_size: int = 1,
    calibrator=None,
    verbose: bool = False
) -> bool:
    """
    Build a TensorRT engine from an ONNX model.

    Args:
        onnx_path     : path to .onnx file
        output_path   : output .engine path
        precision     : 'fp32', 'fp16', or 'int8'
        workspace_gb  : max GPU workspace in GB
        max_batch_size: max batch size for optimization
        calibrator    : INT8Calibrator instance (required for int8)
        verbose       : enable TRT verbose logging

    Returns:
        True if successful
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError(
            "TensorRT not found. Install TensorRT from https://developer.nvidia.com/tensorrt\n"
            "On Jetson devices, TensorRT is pre-installed with JetPack."
        )

    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)

    print(f"\n[TRT Builder] Building {precision.upper()} engine from: {onnx_path}")
    print(f"[TRT Builder] TensorRT version: {trt.__version__}")

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"[TRT Builder] Parse error: {parser.get_error(i)}")
                return False

        print(f"[TRT Builder] ONNX parsed. Inputs: {network.num_inputs}, Outputs: {network.num_outputs}")

        # Builder config
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            workspace_gb * (1 << 30)
        )

        # Precision flags
        if precision == 'fp16':
            if not builder.platform_has_fast_fp16:
                print("[TRT Builder] WARNING: FP16 not natively supported on this GPU.")
            config.set_flag(trt.BuilderFlag.FP16)
            print("[TRT Builder] FP16 mode enabled.")

        elif precision == 'int8':
            if not builder.platform_has_fast_int8:
                print("[TRT Builder] WARNING: INT8 not natively supported on this GPU.")
            config.set_flag(trt.BuilderFlag.INT8)
            config.set_flag(trt.BuilderFlag.FP16)  # FP16 fallback for unsupported layers

            if calibrator is None:
                raise ValueError("INT8 precision requires a calibrator. Pass an INT8Calibrator instance.")
            config.int8_calibrator = calibrator
            print("[TRT Builder] INT8 mode enabled with calibrator.")

        else:
            print("[TRT Builder] FP32 mode (no flags set).")

        # Optimization profile for dynamic shapes (optional)
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        input_shape = network.get_input(0).shape
        h = input_shape[2] if input_shape[2] > 0 else 640
        w = input_shape[3] if input_shape[3] > 0 else 640

        profile.set_shape(
            input_name,
            min=(1, 3, h, w),
            opt=(max_batch_size, 3, h, w),
            max=(max_batch_size, 3, h, w)
        )
        config.add_optimization_profile(profile)

        # Build serialized engine
        print(f"[TRT Builder] Building engine... (this may take 2-10 minutes)")
        t0 = time.time()
        serialized_engine = builder.build_serialized_network(network, config)
        build_time = time.time() - t0

        if serialized_engine is None:
            print("[TRT Builder] ERROR: Engine build failed.")
            return False

        print(f"[TRT Builder] Build completed in {build_time:.1f}s")

        # Save engine
        os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
        with open(output_path, 'wb') as f:
            f.write(serialized_engine)

        size_mb = os.path.getsize(output_path) / (1024 ** 2)
        print(f"[TRT Builder] Engine saved: {output_path} ({size_mb:.1f} MB)")
        return True


class INT8Calibrator:
    """
    TensorRT INT8 Entropy Calibrator.
    Uses a subset of KITTI calibration images to determine quantization scales.
    Recommended: 500-1000 diverse images from validation set.
    """

    def __init__(
        self,
        calib_data_dir: str,
        cache_file: str = 'weights/int8_calib.cache',
        img_size: int = 640,
        batch_size: int = 1,
        max_calib_images: int = 500
    ):
        try:
            import tensorrt as trt
            self.calibration_algorithm = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
        except ImportError:
            pass

        self.calib_dir = Path(calib_data_dir)
        self.cache_file = cache_file
        self.img_size = img_size
        self.batch_size = batch_size

        import glob, cv2
        all_imgs = sorted(glob.glob(str(self.calib_dir / '*.png')))
        all_imgs += sorted(glob.glob(str(self.calib_dir / '*.jpg')))
        self.image_files = all_imgs[:max_calib_images]
        self.batch_idx = 0
        self.max_batches = len(self.image_files) // batch_size

        print(f"[INT8 Calib] Using {len(self.image_files)} images for calibration.")

        import pycuda.driver as cuda
        import pycuda.autoinit
        self.device_input = cuda.mem_alloc(
            batch_size * 3 * img_size * img_size * 4  # float32
        )
        self.cv2 = cv2

    def _preprocess(self, img_path: str) -> np.ndarray:
        img = self.cv2.imread(img_path)
        img = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
        img = self.cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        return np.ascontiguousarray(img)

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names):
        if self.batch_idx >= self.max_batches:
            return None
        batch_start = self.batch_idx * self.batch_size
        batch_imgs = []
        for i in range(self.batch_size):
            idx = batch_start + i
            if idx < len(self.image_files):
                batch_imgs.append(self._preprocess(self.image_files[idx]))

        if not batch_imgs:
            return None

        import pycuda.driver as cuda
        batch = np.stack(batch_imgs, axis=0)
        cuda.memcpy_htod(self.device_input, batch)
        self.batch_idx += 1

        if self.batch_idx % 50 == 0:
            print(f"[INT8 Calib] Calibrated {self.batch_idx}/{self.max_batches} batches")

        return [self.device_input]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"[INT8 Calib] Reading cache: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True) if os.path.dirname(self.cache_file) else None
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
        print(f"[INT8 Calib] Cache saved: {self.cache_file}")


def build_all_precisions(
    onnx_path: str,
    output_dir: str,
    workspace_gb: int = 4,
    calib_data_dir: Optional[str] = None
):
    """Build FP32, FP16, and INT8 engines and print comparison table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    results = {}

    for precision in ['fp32', 'fp16', 'int8']:
        out_path = str(output_dir / f"yolov5_kitti_{precision}.engine")
        calibrator = None

        if precision == 'int8':
            if calib_data_dir is None:
                print("[TRT Builder] Skipping INT8: no calibration data provided.")
                continue
            calibrator = INT8Calibrator(calib_data_dir)

        t0 = time.time()
        success = build_engine(
            onnx_path, out_path, precision, workspace_gb,
            calibrator=calibrator
        )
        build_time = time.time() - t0

        if success:
            size_mb = os.path.getsize(out_path) / (1024 ** 2)
            results[precision] = {'path': out_path, 'size_mb': size_mb, 'build_time': build_time}

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Precision':<12} {'Size (MB)':<14} {'Build Time (s)':<16} {'Engine Path'}")
    print("=" * 60)
    for prec, info in results.items():
        print(f"{prec.upper():<12} {info['size_mb']:<14.1f} {info['build_time']:<16.1f} {info['path']}")
    print("=" * 60)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build TensorRT engines from ONNX')
    parser.add_argument('--onnx',             type=str, required=True)
    parser.add_argument('--output',           type=str, default='weights/')
    parser.add_argument('--precision',        type=str, default='fp16',
                        choices=['fp32', 'fp16', 'int8', 'all'])
    parser.add_argument('--workspace',        type=int, default=4)
    parser.add_argument('--calibration-data', type=str, default=None)
    parser.add_argument('--verbose',          action='store_true')
    args = parser.parse_args()

    if args.precision == 'all':
        build_all_precisions(args.onnx, args.output, args.workspace, args.calibration_data)
    else:
        out = str(Path(args.output) / f"yolov5_kitti_{args.precision}.engine")
        calibrator = None
        if args.precision == 'int8' and args.calibration_data:
            calibrator = INT8Calibrator(args.calibration_data)
        build_engine(args.onnx, out, args.precision, args.workspace, calibrator=calibrator, verbose=args.verbose)
