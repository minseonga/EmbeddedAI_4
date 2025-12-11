"""
Export optimized MobileHand models for Jetson Nano
- ONNX Export
- TensorRT FP16
- Structured Pruning
- INT8 Quantization
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add src to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mobilehand.model import HMR

# Paths
MODEL_DIR = ROOT / "assets/models"
WEIGHTS_PATH = MODEL_DIR / "hmr_model_freihand_auc.pth"
OUTPUT_DIR = MODEL_DIR


def load_mobilehand_model():
    """Load MobileHand HMR model with weights."""
    print("[Load] Initializing MobileHand model...")
    model = HMR(dataset='freihand')
    
    if WEIGHTS_PATH.exists():
        state_dict = torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict)
        print(f"[Load] Loaded weights from {WEIGHTS_PATH}")
    else:
        print(f"[Warning] Weights not found at {WEIGHTS_PATH}")
    
    model.eval()
    return model


def export_onnx(model=None, output_path=None, opset_version=12):
    """
    Export MobileHand to ONNX format.
    
    Args:
        model: HMR model (if None, loads default)
        output_path: Output ONNX file path
        opset_version: ONNX opset version
    """
    print("\n" + "="*60)
    print("ONNX Export")
    print("="*60)
    
    if model is None:
        model = load_mobilehand_model()
    
    if output_path is None:
        output_path = OUTPUT_DIR / "mobilehand.onnx"
    
    model.eval()
    
    # Create dummy input (batch_size=1, 3 channels, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    print(f"[Export] Exporting to {output_path}...")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['keypoints_2d', 'joints_3d', 'vertices', 'angles', 'params'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'keypoints_2d': {0: 'batch_size'},
                'joints_3d': {0: 'batch_size'},
                'vertices': {0: 'batch_size'},
                'angles': {0: 'batch_size'},
                'params': {0: 'batch_size'}
            }
        )
        print(f"[Export] ✓ ONNX export successful: {output_path}")
        print(f"[Export] File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        return output_path
    except Exception as e:
        print(f"[Export] ✗ ONNX export failed: {e}")
        return None


def simplify_onnx(onnx_path):
    """
    Simplify ONNX model using onnxslim.
    """
    try:
        import onnxslim
        print(f"[Simplify] Simplifying {onnx_path}...")
        
        output_path = onnx_path.parent / f"{onnx_path.stem}_opt.onnx"
        onnxslim.slim(str(onnx_path), str(output_path))
        
        print(f"[Simplify] ✓ Simplified: {output_path}")
        return output_path
    except ImportError:
        print("[Simplify] onnxslim not installed. Skipping simplification.")
        return onnx_path
    except Exception as e:
        print(f"[Simplify] ✗ Simplification failed: {e}")
        return onnx_path


def quantize_onnx_int8(onnx_path, output_path=None):
    """
    Apply dynamic INT8 quantization to ONNX model.
    """
    print("\n" + "="*60)
    print("INT8 Quantization (ONNX Dynamic)")
    print("="*60)
    
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        if output_path is None:
            output_path = onnx_path.parent / f"{onnx_path.stem}_int8.onnx"
        
        print(f"[Quantize] Quantizing {onnx_path}...")
        
        quantize_dynamic(
            str(onnx_path),
            str(output_path),
            weight_type=QuantType.QUInt8
        )
        
        original_size = onnx_path.stat().st_size / 1024 / 1024
        quantized_size = output_path.stat().st_size / 1024 / 1024
        reduction = (1 - quantized_size / original_size) * 100
        
        print(f"[Quantize] ✓ INT8 quantization successful: {output_path}")
        print(f"[Quantize] Size: {original_size:.2f} MB → {quantized_size:.2f} MB ({reduction:.1f}% reduction)")
        
        return output_path
    except ImportError:
        print("[Quantize] onnxruntime.quantization not available.")
        return None
    except Exception as e:
        print(f"[Quantize] ✗ Quantization failed: {e}")
        return None


def export_tensorrt_fp16():
    """
    Export to TensorRT FP16 format.
    NOTE: This must be run on the target device (Jetson Nano) with TensorRT installed.
    """
    print("\n" + "="*60)
    print("TensorRT FP16 Export")
    print("="*60)
    
    onnx_path = OUTPUT_DIR / "mobilehand.onnx"
    
    if not onnx_path.exists():
        print("[TensorRT] ONNX model not found. Exporting first...")
        onnx_path = export_onnx()
        if onnx_path is None:
            return None
    
    try:
        import tensorrt as trt
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # Parse ONNX
        with open(str(onnx_path), 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(f"[TensorRT] Parse error: {parser.get_error(error)}")
                return None
        
        # Configure builder
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256MB
        
        # Build engine
        print("[TensorRT] Building FP16 engine (this may take a while)...")
        engine = builder.build_serialized_network(network, config)
        
        if engine is None:
            print("[TensorRT] ✗ Engine build failed")
            return None
        
        # Save engine
        output_path = OUTPUT_DIR / "mobilehand_fp16.engine"
        with open(str(output_path), 'wb') as f:
            f.write(engine)
        
        print(f"[TensorRT] ✓ FP16 engine saved: {output_path}")
        return output_path
        
    except ImportError:
        print("[TensorRT] TensorRT not available. Run this on Jetson Nano.")
        print("[TensorRT] Command on Jetson: python export_mobilehand.py --tensorrt")
        return None
    except Exception as e:
        print(f"[TensorRT] ✗ Export failed: {e}")
        return None


def export_pruned(pruning_ratio=0.3):
    """
    Apply structured pruning to MobileHand encoder (MobileNetV3 only).
    
    The MANO layer has fixed structure and cannot be pruned.
    We prune internal encoder layers while PRESERVING the output dimension (576)
    so the original regressor weights remain compatible.
    
    Args:
        pruning_ratio: Fraction of channels to prune (0.0-1.0)
    """
    print("\n" + "="*60)
    print(f"Structured Pruning - Encoder Only ({int(pruning_ratio*100)}%)")
    print("="*60)
    
    try:
        import torch_pruning as tp
        
        model = load_mobilehand_model()
        
        # Extract encoder only
        encoder = model.encoder
        encoder.eval()
        
        # Count parameters before
        params_before = sum(p.numel() for p in encoder.parameters())
        total_before = sum(p.numel() for p in model.parameters())
        
        print(f"[Pruning] Encoder params before: {params_before:,}")
        
        # Create dummy input for encoder
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Get original encoder output size
        with torch.no_grad():
            original_features = encoder(dummy_input)
            original_size = original_features.shape[1]
            print(f"[Pruning] Original encoder output: {original_size}")
        
        # === KEY: Ignore output layers to preserve 576-dim output ===
        # This keeps the regressor weights compatible
        ignored_layers = []
        for name, module in encoder.named_modules():
            # Skip the final conv layers that produce the 576-dim output
            if 'conv.0' in name or 'conv.1' in name:  # Final conv layers
                ignored_layers.append(module)
            # Also ignore SE (squeeze-excitation) final layers to avoid breaking
            if 'fc.2' in name:
                ignored_layers.append(module)
        
        print(f"[Pruning] Ignoring {len(ignored_layers)} output layers")
        
        # Create pruner for encoder only
        importance = tp.importance.MagnitudeImportance(p=2)
        pruner = tp.pruner.MagnitudePruner(
            encoder,
            example_inputs=dummy_input,
            importance=importance,
            pruning_ratio=pruning_ratio,
            ignored_layers=ignored_layers,
        )
        
        # Apply pruning to encoder
        pruner.step()
        
        # Count parameters after
        params_after = sum(p.numel() for p in encoder.parameters())
        reduction = (1 - params_after / params_before) * 100
        
        print(f"[Pruning] Encoder params after: {params_after:,} ({reduction:.1f}% reduction)")
        
        # Verify encoder output size is preserved
        with torch.no_grad():
            new_features = encoder(dummy_input)
            new_size = new_features.shape[1]
            print(f"[Pruning] New encoder output: {new_size}")
        
        if new_size != original_size:
            print(f"[Warning] Output size changed! Regressor may not work.")
        else:
            print(f"[Pruning] ✓ Output size preserved - regressor compatible!")
        
        # Calculate total model reduction
        total_after = sum(p.numel() for p in model.parameters())
        total_reduction = (1 - total_after / total_before) * 100
        print(f"[Pruning] Total model: {total_before:,} → {total_after:,} ({total_reduction:.1f}% reduction)")
        
        # Verify full model works
        try:
            model.eval()
            with torch.no_grad():
                keypt, joint, vert, ang, params = model(dummy_input)
                print(f"[Pruning] Full model output verified: keypoints shape {keypt.shape}")
        except Exception as e:
            print(f"[Pruning] Warning: Full model verification failed: {e}")
        
        # Save pruned model
        output_path = OUTPUT_DIR / f"mobilehand_pruned_{int(pruning_ratio*100)}.pt"
        torch.save(model.state_dict(), output_path)
        print(f"[Pruning] ✓ Pruned model saved: {output_path}")
        
        # Also export to ONNX
        onnx_path = OUTPUT_DIR / f"mobilehand_pruned_{int(pruning_ratio*100)}.onnx"
        export_onnx(model, onnx_path)
        
        return model, output_path
        
    except ImportError:
        print("[Pruning] torch-pruning not installed.")
        print("[Pruning] Install with: pip install torch-pruning")
        return None, None
    except Exception as e:
        print(f"[Pruning] ✗ Pruning failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def get_model_stats(model):
    """Calculate model statistics."""
    try:
        from thop import profile
        dummy_input = torch.randn(1, 3, 224, 224)
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        return macs / 1e9, params / 1e6  # GFLOPs, M params
    except:
        params = sum(p.numel() for p in model.parameters()) / 1e6
        return 0, params


def main():
    """Main export menu."""
    print("\n" + "="*60)
    print("MobileHand Model Export Tool")
    print("="*60)
    
    # Check if model exists
    if not WEIGHTS_PATH.exists():
        print(f"\n[Error] Model weights not found at {WEIGHTS_PATH}")
        print("Please ensure the MobileHand weights are in place.")
        return
    
    print("\nAvailable export options:")
    print("  1. ONNX Export (basic)")
    print("  2. ONNX + Simplify + INT8 Quantization")
    print("  3. TensorRT FP16 (run on Jetson)")
    print("  4. Structured Pruning (30%)")
    print("  5. Structured Pruning (50%)")
    print("  6. All (ONNX + INT8 + Pruning)")
    print("  0. Exit")
    
    try:
        choice = input("\nSelect option (0-6): ").strip()
    except EOFError:
        choice = "6"  # Default to all
    
    if choice == "0":
        return
    elif choice == "1":
        export_onnx()
    elif choice == "2":
        onnx_path = export_onnx()
        if onnx_path:
            opt_path = simplify_onnx(onnx_path)
            quantize_onnx_int8(opt_path)
    elif choice == "3":
        export_tensorrt_fp16()
    elif choice == "4":
        export_pruned(0.3)
    elif choice == "5":
        export_pruned(0.5)
    elif choice == "6":
        print("\n[All] Running complete export pipeline...")
        
        # 1. Basic ONNX
        onnx_path = export_onnx()
        
        if onnx_path:
            # 2. Simplify
            opt_path = simplify_onnx(onnx_path)
            
            # 3. INT8 Quantization
            quantize_onnx_int8(opt_path)
        
        # 4. Pruning 30%
        export_pruned(0.3)
        
        # 5. Pruning 50%
        export_pruned(0.5)
        
        print("\n" + "="*60)
        print("Export Complete!")
        print("="*60)
        print(f"\nExported models in: {OUTPUT_DIR}")
    else:
        print("[Error] Invalid option")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export MobileHand models")
    parser.add_argument("--onnx", action="store_true", help="Export to ONNX")
    parser.add_argument("--int8", action="store_true", help="ONNX INT8 quantization")
    parser.add_argument("--tensorrt", action="store_true", help="TensorRT FP16 (Jetson)")
    parser.add_argument("--prune", type=float, default=0, help="Pruning ratio (0.0-1.0)")
    parser.add_argument("--all", action="store_true", help="Run all exports")
    
    args = parser.parse_args()
    
    if args.all or not any([args.onnx, args.int8, args.tensorrt, args.prune > 0]):
        main()
    else:
        if args.onnx:
            export_onnx()
        if args.int8:
            onnx_path = OUTPUT_DIR / "mobilehand.onnx"
            if not onnx_path.exists():
                onnx_path = export_onnx()
            if onnx_path:
                opt_path = simplify_onnx(onnx_path)
                quantize_onnx_int8(opt_path)
        if args.tensorrt:
            export_tensorrt_fp16()
        if args.prune > 0:
            export_pruned(args.prune)
