"""
Export YOLOv5 PyTorch model to ONNX format.
First step in the PyTorch → ONNX → TensorRT pipeline.

Usage:
    python optimization/export_onnx.py \
        --weights weights/yolov5_kitti.pt \
        --output weights/yolov5_kitti.onnx \
        --img-size 640 \
        --batch-size 1 \
        --opset 17 \
        --simplify
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


def export_onnx(
    weights: str,
    output: str,
    img_size: int = 640,
    batch_size: int = 1,
    opset: int = 17,
    simplify: bool = True,
    dynamic: bool = False,
    device: str = 'cpu'
) -> str:
    """
    Export YOLOv5 model to ONNX.

    Args:
        weights  : path to .pt weights
        output   : output .onnx path
        img_size : input image size
        batch_size: static batch size (use 1 for TRT)
        opset    : ONNX opset version (17 recommended for TRT 8.6+)
        simplify : run onnx-simplifier for cleaner graph
        dynamic  : enable dynamic batch/spatial axes
        device   : 'cpu' or 'cuda'

    Returns:
        Path to exported .onnx file
    """
    print(f"\n[ONNX Export] Loading weights: {weights}")
    device = torch.device(device)

    # Load model via torch.hub
    model = torch.hub.load(
        'ultralytics/yolov5', 'custom',
        path=weights, force_reload=False
    ).to(device)
    model.eval()

    # Dummy input
    dummy = torch.zeros(batch_size, 3, img_size, img_size).to(device)

    # Dynamic axes config
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            'images': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 1: 'anchors'}
        }

    output_path = str(Path(output))
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

    print(f"[ONNX Export] Exporting to: {output_path}")
    torch.onnx.export(
        model,
        dummy,
        output_path,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )

    # Verify export
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"[ONNX Export] Model check passed.")

    # Simplify graph
    if simplify:
        try:
            import onnxsim
            print("[ONNX Export] Simplifying ONNX graph...")
            simplified, success = onnxsim.simplify(onnx_model)
            if success:
                onnx.save(simplified, output_path)
                print("[ONNX Export] Simplification successful.")
            else:
                print("[ONNX Export] Simplification failed, using original.")
        except ImportError:
            print("[ONNX Export] onnxsim not installed. Skipping simplification.")

    # Print model info
    model_size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"[ONNX Export] Done. File: {output_path} ({model_size_mb:.1f} MB)")
    print(f"[ONNX Export] Input : images [{batch_size}, 3, {img_size}, {img_size}]")
    print(f"[ONNX Export] Opset : {opset}")

    return output_path


def print_onnx_info(onnx_path: str):
    """Print ONNX model graph summary."""
    import onnx
    model = onnx.load(onnx_path)
    print(f"\n=== ONNX Model Info: {onnx_path} ===")
    print(f"IR Version  : {model.ir_version}")
    print(f"Opset       : {model.opset_import[0].version}")
    print(f"Inputs      : {[i.name for i in model.graph.input]}")
    print(f"Outputs     : {[o.name for o in model.graph.output]}")
    print(f"Nodes       : {len(model.graph.node)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export YOLOv5 to ONNX')
    parser.add_argument('--weights',    type=str, required=True)
    parser.add_argument('--output',     type=str, default='weights/yolov5_kitti.onnx')
    parser.add_argument('--img-size',   type=int, default=640)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--opset',      type=int, default=17)
    parser.add_argument('--device',     type=str, default='cpu')
    parser.add_argument('--simplify',   action='store_true', default=True)
    parser.add_argument('--dynamic',    action='store_true', default=False)
    args = parser.parse_args()

    out = export_onnx(
        weights=args.weights,
        output=args.output,
        img_size=args.img_size,
        batch_size=args.batch_size,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=args.dynamic,
        device=args.device
    )
    print_onnx_info(out)
