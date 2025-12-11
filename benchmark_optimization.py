
import time
import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import sys

# Add src to path if needed, though usually automatic if running from root
sys.path.append(str(Path(__file__).parent))
try:
    from src.optimization import get_model_info
except ImportError:
    # Fallback if src is not found (e.g. running from wrong dir)
    print("Warning: could not import src.optimization. FLOPs calculation might fail.")
    def get_model_info(model, input_size=(1,3,640,640)):
        return 0, 0

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "assets/models/yolo11n_hand_pose.pt"
OUTPUT_DIR = ROOT / "assets/models"

class BenchmarkResults:
    def __init__(self):
        self.results = []

    def add(self, name, fps, macs_g, params_m, model_size_mb):
        self.results.append({
            'name': name,
            'fps': fps,
            'macs_g': macs_g,
            'params_m': params_m,
            'model_size_mb': model_size_mb
        })

    def print_table(self):
        print("\n" + "=" * 100)
        print(f"{'Model':<30} {'FPS':<10} {'GFLOPs':<10} {'Params(M)':<10} {'Size(MB)':<10}")
        print("=" * 100)
        for r in self.results:
            print(f"{r['name']:<30} {r['fps']:<10.1f} {r['macs_g']:<10.2f} {r['params_m']:<10.2f} {r['model_size_mb']:<10.2f}")
        print("=" * 100)


def get_model_size_mb(model_path):
    """Get model file size in MB."""
    if not Path(model_path).exists():
        return 0.0
    return Path(model_path).stat().st_size / (1024 * 1024)


def benchmark_inference(model_wrapper, test_frames, num_runs=50):
    """
    Benchmark inference speed.
    Args:
        model_wrapper: YOLO or equivalent callable
        test_frames: list of numpy images
    """
    times = []
    
    # Warmup
    print("  Warmup...", end=" ", flush=True)
    for i in range(1): # Reduced to 1 for Jetson Nano speed
        _ = model_wrapper(test_frames[0], verbose=False)
    print("Done.")

    print("  Benchmarking...", end="\r")
    for i in range(num_runs):
        frame = test_frames[np.random.randint(len(test_frames))]
        start = time.time()
        _ = model_wrapper(frame, verbose=False)
        times.append(time.time() - start)

    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    print(f"  Done. Avg latency: {avg_time*1000:.1f}ms")
    return fps


def prepare_test_frames(num_frames=20):
    """Generate test frames."""
    frames = []
    for _ in range(num_frames):
        # Random 640x480 RGB frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frames.append(frame)
    return frames


def load_custom_pruned_model(path):
    """Load the custom pruned model checkpoint."""
    try:
        if not Path(path).exists():
            return None
        
        # Load checkpoint
        # Compatibility: Removed weights_only=False for older PyTorch versions on Jetson
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
             # Fallback if some other argument issue exists, but standard load should work on old PT
             ckpt = torch.load(path, map_location='cpu')
        
        # If it's a dict with 'model', extract it
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model = ckpt['model']
        else:
            model = ckpt # Assuming it's the model itself
            
        # Wrap in a simple callable to mimic YOLO class behavior for inference
        class PrunedWrapper:
            def __init__(self, model):
                self.model = model
                self.model.eval()
                
            def __call__(self, source, verbose=False):
                # source: numpy image HWC
                # Preprocess
                if isinstance(source, np.ndarray):
                     # Simple resize and normalize for benchmarking speed
                     # This is NOT correct preprocessing for accuracy, but OK for speed benchmark
                     img = cv2.resize(source, (640, 640))
                     img = img.transpose(2, 0, 1)
                     img = torch.from_numpy(img).float() / 255.0
                     img = img.unsqueeze(0)
                else:
                    img = source

                with torch.no_grad():
                    return self.model(img)
                    
        return PrunedWrapper(model), model
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None, None




def benchmark_all():
    import argparse
    parser = argparse.ArgumentParser(description="YOLO Benchmark")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="Path to .pt model")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "baseline", "pruning", "quantization"], 
                        help="Benchmark mode: all, baseline, pruning, or quantization")
    args = parser.parse_args()
    
    model_path = Path(args.model)
    mode = args.mode

    print("=" * 80)
    print(f"YOLO Real Optimization Benchmark (Mode: {mode})")
    print("=" * 80)

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    test_frames = prepare_test_frames()
    results = BenchmarkResults()

    # 1. Baseline
    if mode in ["all", "baseline"]:
        print(f"\n[1] Benchmarking {model_path.name}...")
        
        # Cleanup
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        model_baseline = YOLO(str(model_path))
        print("  Mode potentially FP16 (Automatic)...")
        fps_base = benchmark_inference(model_baseline, test_frames)
        
        try:
            macs_base, params_base = get_model_info(model_baseline.model)
        except:
             macs_base, params_base = 0, 0
        size_base = get_model_size_mb(model_path)
        
        results.add(f"Baseline: {model_path.name}", fps_base, macs_base/1e9, params_base/1e6, size_base)
        
        del model_baseline
        gc.collect()

    # 2. Pruned Models
    if mode in ["all", "pruning"]:
        print("\n[2] Benchmarking Pruned Models (Auto-detected)...")
        pruned_files = sorted(model_path.parent.glob("*_pruned_*.pt"))
        
        for p_path in pruned_files:
            print(f"  Testing {p_path.name}...")
            
            # Cleanup
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            print("    Loading model...", end=" ", flush=True)
            try:
                wrapper, model = load_custom_pruned_model(p_path)
                print("Done.")
            except Exception as e:
                print(f"Failed: {e}")
                continue
            
            if wrapper:
                fps = benchmark_inference(wrapper, test_frames)
                try:
                    macs, params = get_model_info(model)
                except:
                    macs, params = 0, 0
                size = get_model_size_mb(p_path)
                
                rate_str = p_path.stem.split('_')[-1]
                results.add(f"Pruned {rate_str}%", fps, macs/1e9, params/1e6, size)
                
                del wrapper
                del model
                gc.collect()
            else:
                print(f"  Could not load {p_path.name}")

    # 3. Quantized (INT8 via ONNX)
    if mode in ["all", "quantization"]:
        print("\n[3] Benchmarking ONNX/INT8 (Auto-detected)...")
        onnx_files = sorted(model_path.parent.glob("*.onnx"))
        
        for o_path in onnx_files:
            if "fp32" in o_path.name: continue 
            
            print(f"  Testing {o_path.name}...")
            
            # Cleanup
            gc.collect()
            
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                print(f"    Providers: {providers}")
                
                print("    Creating session (First time may take 1-5 mins on Jetson)...", end=" ", flush=True)
                session = ort.InferenceSession(str(o_path), providers=providers)
                print("Done.")
                
                class ONNXWrapper:
                    def __init__(self, sess):
                        self.sess = sess
                        self.input_name = sess.get_inputs()[0].name
                    def __call__(self, source, verbose=False):
                        img = cv2.resize(source, (640, 640))
                        if img.shape[2] == 3: # Ensure C=3
                            img = img.transpose(2, 0, 1)
                        img = img.astype(np.float32) / 255.0
                        img = np.expand_dims(img, 0)
                        self.sess.run(None, {self.input_name: img})

                wrapper = ONNXWrapper(session)
                fps_int8 = benchmark_inference(wrapper, test_frames)
                size_int8 = get_model_size_mb(o_path)
                
                results.add(f"ONNX: {o_path.name}", fps_int8, 0, 0, size_int8)
                
                del session
                del wrapper
                gc.collect()
                
            except ImportError:
                print("onnxruntime not installed.")
            except Exception as e:
                print(f"Failed to benchmark {o_path.name}: {e}")

    results.print_table()

if __name__ == "__main__":
    benchmark_all()

