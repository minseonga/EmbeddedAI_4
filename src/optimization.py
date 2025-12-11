
import torch
import torch_pruning as tp
from ultralytics import YOLO
import torch.nn as nn

class ModelPruner:
    def __init__(self, model_path, example_inputs=None):
        self.model_wrapper = YOLO(model_path)
        self.model = self.model_wrapper.model
        self.example_inputs = example_inputs if example_inputs is not None else torch.randn(1, 3, 640, 640)
        
    def prune(self, pruning_ratio=0.3, importance_method='l1'):
        """
        Apply structured pruning to the model.
        Args:
            pruning_ratio: Target pruning ratio globally (e.g. 0.3 for 30% pruning)
            importance_method: Method to calculate importance ('l1', 'l2', etc.)
        """
        print(f"Pruning model with ratio: {pruning_ratio}")
        
        # 1. Importance criteria
        if importance_method == 'l1':
            imp = tp.importance.MagnitudeImportance(p=1)
        else:
            raise NotImplementedError(f"Importance method {importance_method} not implemented")

        # 2. Pruner initialization
        # Use a global pruning ratio
        ignored_layers = []
        
        # YOLO Head - Ignore output layers logic
        head = self.model.model[-1]
        
        # Try to ignore head outputs responsibly
        if hasattr(head, 'cv2'): 
             for m in head.cv2:
                 last = list(m.modules())[-1]
                 if isinstance(last, torch.nn.Conv2d): ignored_layers.append(last)
        if hasattr(head, 'cv3'):
             for m in head.cv3:
                 last = list(m.modules())[-1]
                 if isinstance(last, torch.nn.Conv2d): ignored_layers.append(last)
        if hasattr(head, 'kpt'):
             for m in head.kpt:
                 last = list(m.modules())[-1]
                 if isinstance(last, torch.nn.Conv2d): ignored_layers.append(last)

        # 3. Create a Tracing Wrapper
        # This is CRITICAL. YOLO's DetectionModel has complex forward logic that TP fails to trace.
        # But the underlying nn.Sequential (self.model.model) is traceable if we wrap it 
        # to accept a single argument properly.
        
        class PruningWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model.model
                self.save = model.save 
            def forward(self, x):
                y = []
                for m in self.model:
                    if m.f != -1:
                        # Replicate the routing logic
                        # m.f is either int or list of ints
                        # DetectionModel logic: x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                        if isinstance(m.f, int):
                            x = y[m.f]
                        else:
                            x = [x if j == -1 else y[j] for j in m.f]
                    x = m(x)
                    y.append(x if m.i in self.save else None)
                return x
        
        wrapper = PruningWrapper(self.model)
        
        # Verify inputs availability
        if self.example_inputs is None:
             self.example_inputs = torch.randn(1, 3, 640, 640)
             
        pruner = tp.pruner.MagnitudePruner(
            wrapper, # Prune the wrapper!
            example_inputs=self.example_inputs,
            importance=imp,
            pruning_ratio=pruning_ratio, 
            ignored_layers=ignored_layers,
            root_module_types=[torch.nn.Conv2d, torch.nn.Linear]
        )

        # 4. Pruning
        base_macs, base_nparams = tp.utils.count_ops_and_params(wrapper, self.example_inputs)
        print(f"Before Pruning:  MACs={base_macs/1e9:.4f} G, Params={base_nparams/1e6:.4f} M")
        print(f"Ignored Layers: {len(ignored_layers)}")

        # Step
        pruner.step()

        # 5. Cleanup & Validation
        pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(wrapper, self.example_inputs)
        print(f"After Pruning:   MACs={pruned_macs/1e9:.4f} G, Params={pruned_nparams/1e6:.4f} M")
        print(f"Reduction:       MACs={1 - pruned_macs/base_macs:.2%}, Params={1 - pruned_nparams/base_nparams:.2%}")

        # IMPORTANT: The 'wrapper.model' is a reference to 'self.model.model'.
        # Since we pruned 'wrapper', 'self.model' is indeed modified in-place.
        # We return the original YOLO wrapper because it holds the config.
        return self.model_wrapper

    def save(self, save_path):
        # We need to save purely the state dict or the full model
        # Ultralytics has its own save mechanism but we modified the internal nn.Module
        # Re-wrapping might be tricky because the config (yaml) doesn't match the new structure.
        # It's safest to save the torch model directly or update the YOLO wrapper.
        
        # For simplicity in this context, we save the weights compatible with torch.load
        # But for Ultralytics optimization, we usually want to save as .pt that YOLO() can load.
        
        # The modified model structure needs to be serialized.
        torch.save(self.model, save_path)
        print(f"Saved pruned model to {save_path}")

def get_model_info(model, input_size=(1,3,640,640)):
    """Return MACs and Params"""
    example_inputs = torch.randn(input_size).to(next(model.parameters()).device)
    macs, params = tp.utils.count_ops_and_params(model, example_inputs)
    return macs, params
