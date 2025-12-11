
import torch
from ultralytics import YOLO
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))
from src.optimization import ModelPruner

def debug_pruning():
    model_path = "assets/models/yolo11n_hand_pose.pt"
    print(f"Loading {model_path}...")
    
    pruner = ModelPruner(model_path)
    
    # Try a 50% prune
    print("Attempting 50% prune with ModelPruner class...")
    pruned_wrapper = pruner.prune(pruning_ratio=0.5)
    
    # Save to a temp file
    temp_path = "debug_model.pt"
    pruner.save(temp_path)
    print(f"Saved to {temp_path}")

if __name__ == "__main__":
    debug_pruning()
        
    print("\n[Method 2] Creating Pruner...")
    
    imp = tp.importance.MagnitudeImportance(p=1)
    
    ignored_layers = []
    
    # Smart ignore logic for YOLO Head
    head = model.model[-1]
    if hasattr(head, 'cv2') and hasattr(head, 'cv3'):
       # YOLOv8/11 Head structure:
       # cv2: list of Conv blocks for box
       # cv3: list of Conv blocks for cls/pose
       # The final layer in each branch determines output size (nc, kpt_shape etc)
       # We must ignore the OUTPUT channels of these final layers.
       # torch_pruning allows pruning input channels of ignored layers usually.
       
       for m in head.cv2:
           # cv2 is a ModuleList of Sequential-like blocks.
           # The last conv in each block is the output (4*reg_max or similar)
           # Let's verify by printing
           # For safety, let's ignore the FINAL conv of each cv2/cv3 branch
           last_layer = list(m.modules())[-1]
           if isinstance(last_layer, torch.nn.Conv2d):
               print(f"  Ignoring Head Output (Box): {type(last_layer)}")
               ignored_layers.append(last_layer)
               
       for m in head.cv3:
           last_layer = list(m.modules())[-1]
           if isinstance(last_layer, torch.nn.Conv2d):
               print(f"  Ignoring Head Output (Cls/Pose): {type(last_layer)}")
               ignored_layers.append(last_layer)

       # For Pose models, there is kpt_head usually inside cv3 or separate.
       # In yolo11-pose, head is 'Pose'.
       # It has self.kpt (Conv2d) if it exists.
       if hasattr(head, 'kpt'):
           for m in head.kpt:
               last_layer = list(m.modules())[-1]
               if isinstance(last_layer, torch.nn.Conv2d):
                   print(f"  Ignoring Head Output (Keypoints): {type(last_layer)}")
                   ignored_layers.append(last_layer)
                   
    print(f"  Total ignored layers: {len(ignored_layers)}")
    
    # Debug: Check Dependency Graph directly
    print("\n[Method 2.5] Building Dependency Graph...")
    
    # Test 1: Full Model
    print("  Testing Full Model (DetectionModel)...")
    dg = tp.DependencyGraph()
    try:
        dg.build_dependency(model, example_inputs=example_inputs)
        print(f"    Nodes: {len(dg.module2node)}")
        if len(dg.module2node) < 10:
             print(f"    Nodes captured: {list(dg.module2node.keys())}")
    except Exception as e:
        print(f"    Failed: {e}")

    # Test 2: Inner Sequential
    print("\n  Testing Inner Sequential (model.model)...")
    dg2 = tp.DependencyGraph()
    try:
        # We need to wrap it because it expects standard args
        dg2.build_dependency(model.model, example_inputs=example_inputs)
        print(f"    Nodes: {len(dg2.module2node)}")
    except Exception as e:
        print(f"    Failed (likely due to shape mismatch in sequential exec): {e}")
    
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=imp,
        pruning_ratio=0.5, # Aggressive 50%
        ignored_layers=ignored_layers,
        iterative_steps=1, 
        root_module_types=[torch.nn.Conv2d, torch.nn.Linear]
    )
    
    print("\n[Method 3] Checking Pruning Plan...")
    try:
        for group in pruner.step(interactive=True):
            print(f"  Group: {group}")
            # If we see things here, it works.
            # If not, it means dependency graph is locking everything.
            break # Just check first group
    except Exception as e:
        print(f"Error during planning: {e}")
        
    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"\nBase MACs: {base_macs/1e9:.3f}G, Params: {base_params/1e6:.3f}M")
    
    pruner.step()
    
    new_macs, new_params = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"New  MACs: {new_macs/1e9:.3f}G, Params: {new_params/1e6:.3f}M")
    print(f"Reduction: {1 - new_macs/base_macs:.2%}")

if __name__ == "__main__":
    debug_pruning()
