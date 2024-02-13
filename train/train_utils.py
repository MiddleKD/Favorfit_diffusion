import os
import torch
from typing import Dict

def update_ckpt_weights(models: Dict, root_ckpt_path: str, dtype=torch.float32) -> Dict:
    if os.path.isdir(root_ckpt_path):
        fns = [os.path.join(root_ckpt_path, cur) for cur in os.listdir(root_ckpt_path)]
    else:
        fns = [root_ckpt_path]
    
    non_match_count = 0
    for model_name in models.keys():
        for fn in fns:
            if model_name in os.path.basename(fn):
                weights = torch.load(fn)
                if isinstance(weights, dict):
                    if isinstance(models[model_name], list):
                        models[model_name][0].load_state_dict(weights, strict=False)
                    else:
                        models[model_name].load_state_dict(weights, strict=False)
                    print(f"{model_name} is updated from ckpt state_dict {fn}")
                elif isinstance(weights, torch.nn.Module):
                    if isinstance(models[model_name], list):
                        models[model_name][0] = weights
                    else:
                        models[model_name] = weights
                    print(f"{model_name} is updated from ckpt pickled nn.modules {fn}")
                else:
                    raise ValueError(f"{fn} is unknown ckpt type")
                if isinstance(models[model_name], list):
                    models[model_name][0].to(dtype=dtype)
                else:
                    models[model_name].to(dtype=dtype)
            else:
                non_match_count += 1
                
    if non_match_count == len(models.keys()):
        raise ValueError(f"Any ckpt is not match for models dict keys")
    
    return models
