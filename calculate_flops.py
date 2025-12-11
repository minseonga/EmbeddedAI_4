"""
Calculate FLOPs/MACs for MobileHand models.
Works for PyTorch models (not ONNX).
"""

import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

MODEL_DIR = ROOT / "assets/models"


def calculate_flops():
    """Calculate FLOPs for all MobileHand variants."""
    print("\n" + "="*70)
    print("MobileHand FLOPs/MACs Calculation")
    print("="*70)
    
    try:
        from thop import profile, clever_format
    except ImportError:
        print("Error: thop not installed. Run: pip install thop")
        return
    
    from mobilehand.model import HMR
    from mobilehand.backbone import mobilenetv3_small
    
    device = 'cpu'  # FLOPs calculation doesn't need GPU
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    results = []
    
    # === 1. Full HMR Model (Baseline) ===
    print("\n[1] Full HMR Model (Baseline)...")
    try:
        model = HMR(dataset='freihand')
        model.eval()
        
        # Use verbose=True to debug
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        macs_g, params_m = macs / 1e9, params / 1e6
        
        results.append(("HMR Full (Baseline)", macs_g, params_m))
        print(f"  MACs: {macs_g:.4f} G")
        print(f"  Params: {params_m:.2f} M")
        del model
    except Exception as e:
        print(f"  Error: {e}")
    
    # === 2. Encoder Only (MobileNetV3) ===
    print("\n[2] Encoder Only (MobileNetV3-Small)...")
    try:
        encoder = mobilenetv3_small()
        encoder.eval()
        
        macs, params = profile(encoder, inputs=(dummy_input,), verbose=False)
        macs_g, params_m = macs / 1e9, params / 1e6
        
        results.append(("Encoder (MobileNetV3)", macs_g, params_m))
        print(f"  MACs: {macs_g:.4f} G")
        print(f"  Params: {params_m:.2f} M")
        del encoder
    except Exception as e:
        print(f"  Error: {e}")
    
    # === 3. Pruned Models ===
    pruned_paths = sorted(MODEL_DIR.glob("mobilehand_pruned_*.pt"))
    for path in pruned_paths:
        prune_ratio = path.stem.split("_")[-1]
        print(f"\n[3] Pruned {prune_ratio}% Model...")
        
        try:
            # For pruned models, we need to recreate the structure
            # Since encoder channels changed, we measure what we can
            model = HMR(dataset='freihand')
            
            # Try to load - this will fail due to shape mismatch
            # but we can still estimate from the saved state dict
            state_dict = torch.load(path, map_location='cpu', weights_only=False)
            
            # Count actual parameters in saved model
            total_params = sum(v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor))
            params_m = total_params / 1e6
            
            # Estimate MACs reduction (proportional to params for conv layers)
            baseline_params = 3.38  # M
            reduction = params_m / baseline_params
            estimated_macs = 0.06 * reduction  # G
            
            results.append((f"Pruned {prune_ratio}%", estimated_macs, params_m))
            print(f"  Params: {params_m:.2f} M")
            print(f"  Estimated MACs: {estimated_macs:.4f} G (based on param reduction)")
            
            del model, state_dict
        except Exception as e:
            print(f"  Error: {e}")
    
    # === Print Summary ===
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Model':<30} {'MACs (G)':<15} {'Params (M)':<15}")
    print("-"*70)
    
    for name, macs, params in results:
        print(f"{name:<30} {macs:<15.4f} {params:<15.2f}")
    
    print("="*70)
    print("\nNote: MACs ≈ 0.5 × FLOPs (each MAC = 1 multiply + 1 add)")
    print("      FLOPs = MACs × 2")


def calculate_onnx_flops(onnx_path):
    """Calculate FLOPs for ONNX model using onnx-opcounter."""
    print(f"\n[ONNX] Calculating FLOPs for {onnx_path}...")
    
    try:
        import onnx
        from onnx_opcounter import calculate_params, calculate_macs
        
        model = onnx.load(str(onnx_path))
        
        # Calculate params
        params = calculate_params(model)
        params_m = params / 1e6
        
        # Calculate MACs
        macs = calculate_macs(model)
        macs_g = macs / 1e9
        flops_g = macs_g * 2
        
        print(f"\n  Results for: {Path(onnx_path).name}")
        print(f"  ----------------------------------------")
        print(f"  Parameters: {params_m:.2f} M")
        print(f"  MACs:       {macs_g:.4f} G")
        print(f"  FLOPs:      {flops_g:.4f} G")
        print(f"  File Size:  {Path(onnx_path).stat().st_size / 1024 / 1024:.2f} MB")
        
        return macs_g, params_m
        
    except ImportError as e:
        print(f"  Error: {e}")
        print("  Install with: pip install onnx-opcounter")
        return 0, 0
    except Exception as e:
        print(f"  Error: {e}")
        return 0, 0


def calculate_all_onnx():
    """Calculate FLOPs for all ONNX models."""
    print("\n" + "="*70)
    print("ONNX FLOPs Calculation")
    print("="*70)
    
    onnx_files = sorted(MODEL_DIR.glob("mobilehand*.onnx"))
    
    if not onnx_files:
        print("No ONNX files found in assets/models/")
        return
    
    results = []
    for path in onnx_files:
        macs, params = calculate_onnx_flops(path)
        if macs > 0:
            results.append((path.name, macs * 2, params, path.stat().st_size / 1024 / 1024))
    
    # Print summary
    if results:
        print("\n" + "="*70)
        print("ONNX SUMMARY")
        print("="*70)
        print(f"{'Model':<35} {'FLOPs (G)':<12} {'Params (M)':<12} {'Size (MB)':<10}")
        print("-"*70)
        for name, flops, params, size in results:
            print(f"{name:<35} {flops:<12.4f} {params:<12.2f} {size:<10.2f}")
        print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate FLOPs for MobileHand")
    parser.add_argument("--onnx", type=str, help="Calculate FLOPs for specific ONNX model")
    parser.add_argument("--all-onnx", action="store_true", help="Calculate FLOPs for all ONNX models")
    parser.add_argument("--pytorch", action="store_true", help="Calculate FLOPs for PyTorch models")
    
    args = parser.parse_args()
    
    if args.onnx:
        calculate_onnx_flops(Path(args.onnx))
    elif args.all_onnx:
        calculate_all_onnx()
    elif args.pytorch:
        calculate_flops()
    else:
        # Default: both
        calculate_flops()
        calculate_all_onnx()
