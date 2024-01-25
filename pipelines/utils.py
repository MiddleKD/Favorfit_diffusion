import torch

def rescale(x, old_range, new_range, clamp=False):
    x = x.to(dtype=torch.float16)

    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep, dtype=torch.float16):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=dtype) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=dtype)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

def get_model_weights_dtypes(models_wrapped_dict, verbose=False):
    
    model_dtype_map = {}
    for name, model in models_wrapped_dict.items():
        first_param = next(model.parameters())
        model_dtype_map[name] = first_param.dtype
    
    dtype_list = list(model_dtype_map.values())
    
    if verbose == True:
        if all(x == dtype_list[0] for x in dtype_list):
            print(f"Models have same precision")
        else:
            print(f"Models are mixed precision")
            print(model_dtype_map)

    return model_dtype_map
