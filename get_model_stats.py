
import torch
import torch.nn as nn
from pathlib import Path
from ultralytics import YOLO
import sys
import gc

# Add src to path
sys.path.append(str(Path(__file__).parent))
try:
    from src.optimization import get_model_info, ModelPruner
except ImportError:
    print("Warning: could not import src.optimization.")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "assets/models/yolo11n_hand_pose.pt"

def get_model_size_mb(model_path):
    if not Path(model_path).exists(): return 0.0
    return Path(model_path).stat().st_size / (1024 * 1024)

def check_model_stats():
    print("=" * 80)
    print("YOLO Model Stats (No Inference Benchmark)")
    print("=" * 80)
    print(f"{'Model':<40} {'GFLOPs':<10} {'Params(M)':<10} {'Size(MB)':<10}")
    print("-" * 80)

    # 1. Baseline
    try:
        model = YOLO(str(MODEL_PATH)).model
        macs, params = get_model_info(model)
        size = get_model_size_mb(MODEL_PATH)
        print(f"{MODEL_PATH.name:<40} {macs/1e9:<10.2f} {params/1e6:<10.2f} {size:<10.2f}")
    except Exception as e:
        print(f"{MODEL_PATH.name:<40} Error: {e}")

    # 2. Pruned Models
    pruned_files = sorted(MODEL_PATH.parent.glob("*_pruned_*.pt"))
    for p_path in pruned_files:
        try:
            # We need to load it properly to count MACs
            # Trusted source: weights_only=False
            ckpt = torch.load(p_path, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict) and 'model' in ckpt:
                model = ckpt['model']
            else:
                model = ckpt
            
            # Use get_model_info which handles dummy input
            macs, params = get_model_info(model)
            size = get_model_size_mb(p_path)
            print(f"{p_path.name:<40} {macs/1e9:<10.2f} {params/1e6:<10.2f} {size:<10.2f}")
            
            del model
            del ckpt
            gc.collect()
        except Exception as e:
            print(f"{p_path.name:<40} Error: {e}")

    # 3. ONNX Models (Size only, MACs hard to count without external tools)
    onnx_files = sorted(MODEL_PATH.parent.glob("*.onnx"))
    for o_path in onnx_files:
        size = get_model_size_mb(o_path)
        # MACs for ONNX is not trivial with standard torch tools, usually 0 or N/A
        print(f"{o_path.name:<40} {'-':<10} {'-':<10} {size:<10.2f}")

    print("=" * 80)
    print("Use 'python src/emoji_reactor/app.py' to test FPS and latency.")

if __name__ == "__main__":
    check_model_stats()
