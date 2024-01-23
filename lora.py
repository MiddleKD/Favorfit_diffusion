import torch.nn as nn

def get_lora_layers(in_features, out_features, rank=4):
    lora_down = nn.Linear(in_features, rank, bias=False)
    lora_up = nn.Linear(rank, out_features, bias=False)
    return lora_down, lora_up


class LoraWrapper(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        for key, value in state_dict.items():
            setattr(self, key, value)


def get_specific_key_modules(parent_module, target_key):

    selected_modules_dict = {}

    def recursice_named_children(name, cur_module):
        for subname, module in cur_module.named_children():
            recursice_named_children(f"{name}.{subname}",module)
            if f"{target_key}" in subname:
                selected_modules_dict[f"{name}.{subname}"] = module
    
    for name, cur_module in parent_module.named_children():
        recursice_named_children(name, cur_module)

    return selected_modules_dict


def extract_lora_from_unet(unet):
    selected_modules_dict = get_specific_key_modules(unet, "lora")
    print("lora keys seleted num: ", len(selected_modules_dict))
    lora_wrapper_model = LoraWrapper(selected_modules_dict)
    return lora_wrapper_model

