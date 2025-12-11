"""
Export optimized models for Jetson Nano
- TensorRT INT8 quantization
- TensorRT FP16
- Structured pruning
"""

import torch
import torch.nn as nn
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "assets/models/yolo11n_hand_pose.pt"
OUTPUT_DIR = ROOT / "assets/models"


def export_tensorrt_int8():
    """
    Export YOLO11n to TensorRT INT8 format.
    Requires calibration data for INT8 quantization.
    """
    print("\n" + "=" * 80)
    print("Exporting TensorRT INT8 Model")
    print("=" * 80)

    model = YOLO(str(MODEL_PATH))

    print("\nExporting to TensorRT INT8...")
    print("This may take several minutes...")

    try:
        # Export to TensorRT with INT8
        model.export(
            format='engine',
            half=False,
            int8=True,
            imgsz=640,
            device=0  # Use GPU 0
        )

        engine_path = MODEL_PATH.with_suffix('.engine')
        int8_path = OUTPUT_DIR / "yolo11n_hand_pose_int8.engine"

        if engine_path.exists():
            engine_path.rename(int8_path)
            print(f"\n✓ INT8 TensorRT engine saved: {int8_path}")
            print(f"  Size: {int8_path.stat().st_size / (1024*1024):.2f} MB")
        else:
            print("\n✗ Export failed - TensorRT engine not found")

    except Exception as e:
        print(f"\n✗ INT8 export failed: {e}")
        print("\nNote: INT8 quantization requires:")
        print("  - NVIDIA GPU with CUDA")
        print("  - TensorRT installed")
        print("  - pycuda installed")
        print("\nFor Jetson Nano, run this script on the device itself.")


def export_tensorrt_fp16():
    """
    Export YOLO11n to TensorRT FP16 format.
    FP16 provides good speed/accuracy trade-off on Jetson Nano.
    """
    print("\n" + "=" * 80)
    print("Exporting TensorRT FP16 Model")
    print("=" * 80)

    model = YOLO(str(MODEL_PATH))

    print("\nExporting to TensorRT FP16...")
    print("This may take a few minutes...")

    try:
        # Export to TensorRT with FP16
        model.export(
            format='engine',
            half=True,  # FP16
            imgsz=640,
            device=0
        )

        engine_path = MODEL_PATH.with_suffix('.engine')
        fp16_path = OUTPUT_DIR / "yolo11n_hand_pose_fp16.engine"

        if engine_path.exists():
            engine_path.rename(fp16_path)
            print(f"\n✓ FP16 TensorRT engine saved: {fp16_path}")
            print(f"  Size: {fp16_path.stat().st_size / (1024*1024):.2f} MB")
        else:
            print("\n✗ Export failed - TensorRT engine not found")

    except Exception as e:
        print(f"\n✗ FP16 export failed: {e}")
        print("\nNote: FP16 export requires:")
        print("  - NVIDIA GPU with CUDA")
        print("  - TensorRT installed")



def export_pruned_models():
    """
    Export pruned models at different pruning rates using structured pruning.
    """
    print("\n" + "=" * 80)
    print("Exporting Pruned Models (Structured Pruning)")
    print("=" * 80)

    # Import locally to avoid issues if requirements aren't met yet
    try:
        from src.optimization import ModelPruner
    except ImportError as e:
        print(f"Error importing ModelPruner: {e}")
        return

    pruning_rates = [0.1, 0.3, 0.5, 0.7]

    for rate in pruning_rates:
        print(f"\n[Pruning Rate: {rate*100:.0f}%]")

        # Load fresh model wrapper
        pruner = ModelPruner(str(MODEL_PATH))

        print(f"  Applying {rate*100:.0f}% structured pruning...")
        # Apply pruning
        pruned_wrapper = pruner.prune(pruning_ratio=rate)
        
        # Save pruned model
        pruned_path = OUTPUT_DIR / f"yolo11n_hand_pose_pruned_{int(rate*100)}.pt"

        try:
            # We save the dictionary format that YOLO expects/is compatible with
            # torch.save(pruned_wrapper.model, str(pruned_path)) would save just the model
            # typical ultralytics .pt has {'model': ...}
            
            # Create a simplified checkpoint
            ckpt = {'model': pruned_wrapper.model}
            torch.save(ckpt, str(pruned_path))

            print(f"  ✓ Saved: {pruned_path}")
            print(f"    Size: {pruned_path.stat().st_size / (1024*1024):.2f} MB")

        except Exception as e:
            print(f"  ✗ Failed to save: {e}")



def export_onnx():
    """
    Export to ONNX format for maximum compatibility.
    """
    print("\n" + "=" * 80)
    print("Exporting ONNX Model")
    print("=" * 80)

    model = YOLO(str(MODEL_PATH))

    print("\nExporting to ONNX...")

    try:
        model.export(
            format='onnx',
            dynamic=False,
            simplify=True,
            opset=12,
            imgsz=640
        )

        onnx_path = MODEL_PATH.with_suffix('.onnx')
        final_path = OUTPUT_DIR / "yolo11n_hand_pose.onnx"

        if onnx_path.exists():
            if final_path != onnx_path:
                if final_path.exists():
                    final_path.unlink()
                onnx_path.rename(final_path)
            print(f"\n✓ ONNX model saved: {final_path}")
            print(f"  Size: {final_path.stat().st_size / (1024*1024):.2f} MB")
            return final_path
        else:
            print("\n✗ Export failed - ONNX file not found")
            return None

    except Exception as e:
        print(f"\n✗ ONNX export failed: {e}")
        return None

def quantize_onnx_model(onnx_path):
    """
    Apply Dynamic Quantization to an ONNX model.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        qt_path = onnx_path.parent / f"{onnx_path.stem}_int8.onnx"
        print(f"  Quantizing {onnx_path.name} to INT8...")
        
        quantize_dynamic(
            str(onnx_path),
            str(qt_path),
            weight_type=QuantType.QUInt8
        )
        print(f"    ✓ Saved INT8 model: {qt_path.name}")
        return qt_path
    except ImportError:
        print("    Warning: onnxruntime not installed, skipping quantization")
        return None
    except Exception as e:
        print(f"    Warning: Quantization failed: {e}")
        return None


def export_pruned_quantized():
    """
    Load pruned models, export to ONNX, then Quantize (INT8).
    This creates the "Pruned + Quantized" pipeline.
    """
    print("\n" + "=" * 80)
    print("Exporting Pruned + Quantized Models (INT8)")
    print("=" * 80)
    
    # Check for pruned files
    pruned_files = sorted(OUTPUT_DIR.glob("*_pruned_*.pt"))
    if not pruned_files:
        print("No pruned models found. Please run Option 3 (Pruning) first.")
        # Optional: Run pruning here implicitly? For now, let's keep it explicit.
        return

    for p_path in pruned_files:
        print(f"\nProcessing {p_path.name}...")
        
        # 1. Load Pruned Model
        # We need to wrap it so Ultralytics YOLO can export it
        import copy
        try:
            # Load custom dict
            # Compatibility: Removed weights_only argument
            ckpt = torch.load(p_path, map_location='cpu')
            if isinstance(ckpt, dict) and 'model' in ckpt:
                model_obj = ckpt['model']
            else:
                model_obj = ckpt
            
            # Create a YOLO wrapper for export
            # This is tricky because YOLO() expects a file path usually.
            # We can try to load a dummy and swap weights, OR use the export API directly if possible.
            # Easiest way: re-save the pruned model as a temp .pt that YOLO() accepts?
            # Actually, we can assign the model to a YOLO instance.
            
            yolo_wrapper = YOLO(str(MODEL_PATH)) # Load basline just to get structure
            yolo_wrapper.model = model_obj # Swap with pruned model
            
            # 2. Export to ONNX
            # Note: We need a unique name for the temporary/final ONNX
            # Ultralytics export usually uses the source filename.
            # Let's see if we can trick it or simpler: use the torch.onnx directly?
            # Ultralytics export handles post-processing ops which is vital.
            
            # Trick: Save pruned model to a temp path that reflects its name?
            # Actually, `yolo_wrapper.export` might use the `model` attribute's internal logic.
            # But the file naming is derived from .pt path usually.
            
            print("  Exporting pruned ONNX...")
            # We will use the generic ONNX export logic but we need to supply the model object
            # Ultralytics 8.x: model.export()
            
            # Temporarily save pruned model to a clean path so export generates correct name
            temp_pt_path = OUTPUT_DIR / p_path.name
            if not temp_pt_path.exists():
                torch.save(ckpt, temp_pt_path)
            
            temp_model = YOLO(str(temp_pt_path))
            temp_model.export(
                format='onnx',
                dynamic=False,
                simplify=True,
                opset=12,
                imgsz=640
            )
            
            # Resulting ONNX
            onnx_path = temp_pt_path.with_suffix('.onnx')
            
            if onnx_path.exists():
                # 3. Quantize the ONNX
                quantize_onnx_model(onnx_path)
            else:
                print("  Export failed to generate ONNX.")

        except Exception as e:
            print(f"  Failed Pruned+Quantized export: {e}")


def main():
    print("=" * 80)
    print("YOLO11n Hand Pose - Model Optimization Export")
    print("=" * 80)

    if not MODEL_PATH.exists():
        print(f"\n✗ Error: Model not found at {MODEL_PATH}")
        return
    
    # Ensure src package is importable
    import sys
    sys.path.append(str(ROOT))

    print(f"\nSource model: {MODEL_PATH}")
    print(f"Size: {MODEL_PATH.stat().st_size / (1024*1024):.2f} MB")

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\nSelect export format:")
    print("  1) TensorRT INT8 (best speed, ~2% accuracy loss)")
    print("  2) TensorRT FP16 (good speed, minimal accuracy loss)")
    print("  3) Pruned models (Structured Pruning)")
    print("  4) ONNX (FP32/INT8)")
    print("  5) Pruned + Quantized (Pruning -> ONNX INT8)")
    print("  6) All of the above")

    try:
        choice = input("\nEnter choice (1-6): ").strip()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        return

    if choice == '1':
        export_tensorrt_int8()
    elif choice == '2':
        export_tensorrt_fp16()
    elif choice == '3':
        export_pruned_models()
    elif choice == '4':
        path = export_onnx()
        if path: quantize_onnx_model(path)
    elif choice == '5':
        export_pruned_quantized()
    elif choice == '6':
        export_onnx() # Base ONNX
        export_pruned_models() # Pruned PT
        export_pruned_quantized() # Pruned ONNX + INT8
        
        # TensorRT (Might fail on Mac)
        export_tensorrt_fp16()
        export_tensorrt_int8()
    else:
        print("Invalid choice")
        return

    print("\n" + "=" * 80)
    print("Export Complete!")
    print("=" * 80)
    print("\nTo use the optimized models:")
    print("  INT8: python src/emoji_reactor/app.py --precision int8")
    print("  FP16: python src/emoji_reactor/app.py --precision fp16")
    print("  FP32: python src/emoji_reactor/app.py --precision fp32")
    print("\nNote: TensorRT models (.engine) are platform-specific.")
    print("      Export INT8/FP16 on the target device (Jetson Nano).")
    print("=" * 80)


if __name__ == "__main__":
    main()
