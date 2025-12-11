"""
Benchmark MobileHand models
- PyTorch baseline
- ONNX Runtime
- Pruned models
- Quantized models
"""

import time
import torch
import numpy as np
from pathlib import Path
import sys
import gc
import argparse

# Add src to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

MODEL_DIR = ROOT / "assets/models"


class BenchmarkResults:
    """Store and display benchmark results."""
    
    def __init__(self):
        self.results = []
    
    def add(self, name, fps, params_m=0, model_size_mb=0, macs_g=0):
        self.results.append({
            'name': name,
            'fps': fps,
            'params_m': params_m,
            'size_mb': model_size_mb,
            'macs_g': macs_g
        })
    
    def print_table(self):
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80)
        print(f"{'Model':<35} {'FPS':>10} {'Params(M)':>12} {'Size(MB)':>10} {'GFLOPs':>10}")
        print("-"*80)
        
        for r in self.results:
            print(f"{r['name']:<35} {r['fps']:>10.1f} {r['params_m']:>12.2f} {r['size_mb']:>10.2f} {r['macs_g']:>10.2f}")
        
        print("="*80)


def get_model_size_mb(path):
    """Get file size in MB."""
    if path.exists():
        return path.stat().st_size / 1024 / 1024
    return 0


def prepare_test_inputs(num_samples=50):
    """Prepare test inputs for benchmarking."""
    # Random images simulating hand crops (224x224)
    inputs = []
    for _ in range(num_samples):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        inputs.append(img)
    return inputs


def benchmark_pytorch(model, test_inputs, device='cpu', num_warmup=10, num_runs=50):
    """
    Benchmark PyTorch model inference.
    """
    model.to(device)
    model.eval()
    
    # Prepare tensor inputs
    tensors = []
    for img in test_inputs[:num_runs]:
        t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        t = t.unsqueeze(0).to(device)
        tensors.append(t)
    
    # Warmup
    with torch.no_grad():
        for i in range(min(num_warmup, len(tensors))):
            _ = model(tensors[i % len(tensors)])
    
    # Benchmark
    if device == 'cuda':
        torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for t in tensors:
            start = time.perf_counter()
            _ = model(t)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    return fps


def benchmark_onnx(onnx_path, test_inputs, num_warmup=10, num_runs=50):
    """
    Benchmark ONNX Runtime inference.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ONNX] onnxruntime not installed")
        return 0
    
    # Create session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        sess = ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception as e:
        print(f"[ONNX] Failed to load {onnx_path}: {e}")
        return 0
    
    input_name = sess.get_inputs()[0].name
    
    # Prepare inputs
    inputs = []
    for img in test_inputs[:num_runs]:
        t = img.astype(np.float32).transpose(2, 0, 1) / 255.0
        t = np.expand_dims(t, axis=0)
        inputs.append(t)
    
    # Warmup
    for i in range(min(num_warmup, len(inputs))):
        _ = sess.run(None, {input_name: inputs[i % len(inputs)]})
    
    # Benchmark
    times = []
    for inp in inputs:
        start = time.perf_counter()
        _ = sess.run(None, {input_name: inp})
        times.append(time.perf_counter() - start)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    return fps


def benchmark_tensorrt(engine_path, test_inputs, num_warmup=10, num_runs=50):
    """
    Benchmark TensorRT engine inference.
    NOTE: Must be run on device with TensorRT installed (e.g., Jetson Nano)
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        print("[TensorRT] TensorRT/PyCUDA not available")
        return 0
    
    # Load engine
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(str(engine_path), 'rb') as f:
        engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # Allocate buffers
    inputs_buf = []
    outputs_buf = []
    bindings = []
    
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        
        if engine.binding_is_input(binding):
            inputs_buf.append({'host': host_mem, 'device': device_mem})
        else:
            outputs_buf.append({'host': host_mem, 'device': device_mem})
    
    stream = cuda.Stream()
    
    # Prepare inputs
    inputs = []
    for img in test_inputs[:num_runs]:
        t = img.astype(np.float32).transpose(2, 0, 1) / 255.0
        inputs.append(t.flatten())
    
    # Warmup
    for i in range(min(num_warmup, len(inputs))):
        np.copyto(inputs_buf[0]['host'], inputs[i % len(inputs)])
        cuda.memcpy_htod_async(inputs_buf[0]['device'], inputs_buf[0]['host'], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        stream.synchronize()
    
    # Benchmark
    times = []
    for inp in inputs:
        np.copyto(inputs_buf[0]['host'], inp)
        cuda.memcpy_htod_async(inputs_buf[0]['device'], inputs_buf[0]['host'], stream)
        
        start = time.perf_counter()
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        stream.synchronize()
        times.append(time.perf_counter() - start)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    return fps


def get_model_stats(model):
    """Get model parameters and FLOPs."""
    params = sum(p.numel() for p in model.parameters()) / 1e6
    
    try:
        from thop import profile
        dummy = torch.randn(1, 3, 224, 224)
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        macs_g = macs / 1e9
    except:
        macs_g = 0
    
    return params, macs_g


def benchmark_all(mode='all'):
    """
    Run all benchmarks.
    
    Args:
        mode: 'all', 'pytorch', 'onnx', or 'tensorrt'
    """
    print("\n" + "="*60)
    print("MobileHand Benchmark")
    print("="*60)
    
    results = BenchmarkResults()
    test_inputs = prepare_test_inputs(50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # === PyTorch Baseline ===
    if mode in ['all', 'pytorch']:
        print("\n[1/5] PyTorch Baseline...")
        try:
            from mobilehand.model import HMR
            
            model = HMR(dataset='freihand')
            weights_path = MODEL_DIR / "hmr_model_freihand_auc.pth"
            
            if weights_path.exists():
                model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=False))
            
            params, macs_g = get_model_stats(model)
            fps = benchmark_pytorch(model, test_inputs, device)
            size_mb = get_model_size_mb(weights_path)
            
            results.add("PyTorch Baseline", fps, params, size_mb, macs_g)
            print(f"  FPS: {fps:.1f}")
            
            del model
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"  Error: {e}")
    
    # === ONNX FP32 ===
    if mode in ['all', 'onnx']:
        print("\n[2/5] ONNX FP32...")
        onnx_path = MODEL_DIR / "mobilehand.onnx"
        if onnx_path.exists():
            fps = benchmark_onnx(onnx_path, test_inputs)
            size_mb = get_model_size_mb(onnx_path)
            results.add("ONNX FP32", fps, 0, size_mb, 0)
            print(f"  FPS: {fps:.1f}")
        else:
            print(f"  Not found: {onnx_path}")
    
    # === ONNX INT8 ===
    if mode in ['all', 'onnx']:
        print("\n[3/5] ONNX INT8...")
        int8_paths = list(MODEL_DIR.glob("mobilehand*_int8.onnx"))
        for path in int8_paths:
            fps = benchmark_onnx(path, test_inputs)
            size_mb = get_model_size_mb(path)
            results.add(f"ONNX INT8 ({path.stem})", fps, 0, size_mb, 0)
            print(f"  {path.name}: {fps:.1f} FPS")
        
        if not int8_paths:
            print("  Not found")
    
    # === Pruned Models ===
    if mode in ['all', 'pytorch', 'onnx']:
        print("\n[4/5] Pruned Models...")
        
        # PyTorch pruned
        pruned_paths = list(MODEL_DIR.glob("mobilehand_pruned_*.pt"))
        for path in pruned_paths:
            try:
                from mobilehand.model import HMR
                model = HMR(dataset='freihand')
                model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False))
                
                params, macs_g = get_model_stats(model)
                fps = benchmark_pytorch(model, test_inputs, device)
                size_mb = get_model_size_mb(path)
                
                name = path.stem.replace("mobilehand_", "")
                results.add(f"PyTorch {name}", fps, params, size_mb, macs_g)
                print(f"  {path.name}: {fps:.1f} FPS")
                
                del model
                gc.collect()
            except Exception as e:
                print(f"  Error loading {path}: {e}")
        
        # ONNX pruned
        pruned_onnx = list(MODEL_DIR.glob("mobilehand_pruned_*.onnx"))
        for path in pruned_onnx:
            if '_int8' not in path.name:
                fps = benchmark_onnx(path, test_inputs)
                size_mb = get_model_size_mb(path)
                name = path.stem.replace("mobilehand_", "")
                results.add(f"ONNX {name}", fps, 0, size_mb, 0)
                print(f"  {path.name}: {fps:.1f} FPS")
    
    # === TensorRT ===
    if mode in ['all', 'tensorrt']:
        print("\n[5/5] TensorRT...")
        engine_paths = list(MODEL_DIR.glob("mobilehand*.engine"))
        for path in engine_paths:
            fps = benchmark_tensorrt(path, test_inputs)
            size_mb = get_model_size_mb(path)
            name = path.stem.replace("mobilehand_", "")
            results.add(f"TensorRT {name}", fps, 0, size_mb, 0)
            print(f"  {path.name}: {fps:.1f} FPS")
        
        if not engine_paths:
            print("  No TensorRT engines found")
            print("  Run export_mobilehand.py --tensorrt on Jetson Nano")
    
    # Print summary
    results.print_table()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark MobileHand models")
    parser.add_argument("--mode", choices=['all', 'pytorch', 'onnx', 'tensorrt'], 
                        default='all', help="Benchmark mode")
    
    args = parser.parse_args()
    benchmark_all(args.mode)


if __name__ == "__main__":
    main()
