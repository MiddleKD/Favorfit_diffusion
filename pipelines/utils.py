import json
import torch
import numpy as np
from PIL import Image
from typing import Union

def prepare_latent_width_height(pil_image_list=None, explicitly_define_size:Union[list, None]=None, vae_scale=8):
    if explicitly_define_size:
        WIDTH, HEIGHT = explicitly_define_size
        LATENTS_WIDTH = (WIDTH // vae_scale) + (-(WIDTH // vae_scale) % vae_scale if (WIDTH // vae_scale) % vae_scale < vae_scale/2 else (vae_scale - (WIDTH // vae_scale) % vae_scale))
        LATENTS_HEIGHT = (HEIGHT // vae_scale) + (-(HEIGHT // vae_scale) % vae_scale if (HEIGHT // vae_scale) % vae_scale < vae_scale/2 else (vae_scale - (HEIGHT // vae_scale) % vae_scale))
        return WIDTH, HEIGHT, LATENTS_WIDTH*vae_scale, LATENTS_HEIGHT*vae_scale, LATENTS_WIDTH, LATENTS_HEIGHT

    image_sizes = []
    for pil_image in pil_image_list:
        if isinstance(pil_image, list):
            for cur in pil_image:
                if pil_image is not None:
                    image_sizes.append(cur.size)
        else:
            if pil_image is not None:
                image_sizes.append(pil_image.size)
    
    if len(image_sizes) == 0:
        WIDTH, HEIGHT = (512, 512)
        LATENTS_WIDTH = (WIDTH // vae_scale) + (-(WIDTH // vae_scale) % vae_scale if (WIDTH // vae_scale) % vae_scale < vae_scale/2 else (vae_scale - (WIDTH // vae_scale) % vae_scale))
        LATENTS_HEIGHT = (HEIGHT // vae_scale) + (-(HEIGHT // vae_scale) % vae_scale if (HEIGHT // vae_scale) % vae_scale < vae_scale/2 else (vae_scale - (HEIGHT // vae_scale) % vae_scale))
        return WIDTH, HEIGHT, LATENTS_WIDTH*vae_scale, LATENTS_HEIGHT*vae_scale, LATENTS_WIDTH, LATENTS_HEIGHT

    if not all(size == image_sizes[0] for size in image_sizes):
        raise ValueError("All image must have same size")
            
    WIDTH, HEIGHT = image_sizes[0]
    LATENTS_WIDTH = (WIDTH // vae_scale) + (-(WIDTH // vae_scale) % vae_scale if (WIDTH // vae_scale) % vae_scale < vae_scale/2 else (vae_scale - (WIDTH // vae_scale) % vae_scale))
    LATENTS_HEIGHT = (HEIGHT // vae_scale) + (-(HEIGHT // vae_scale) % vae_scale if (HEIGHT // vae_scale) % vae_scale < vae_scale/2 else (vae_scale - (HEIGHT // vae_scale) % vae_scale))

    return WIDTH, HEIGHT, LATENTS_WIDTH*vae_scale, LATENTS_HEIGHT*vae_scale, LATENTS_WIDTH, LATENTS_HEIGHT

def check_prompt_text_length(prompt_list, max_length=77):
    truncated_prompt_list = []
    for prompt in prompt_list:
        if isinstance(prompt, str):
            if len(prompt) > max_length:
                print(f"prompts is too long it will be truncated to {max_length} len")
            truncated_prompt_list.append(prompt[:max_length])
        else:
            truncated_prompt_list.append(prompt)
    return truncated_prompt_list

def rescale(x, old_range, new_range, clamp=False, out_type="pt"):
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)
    
    x = x.to(dtype=torch.float16)

    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)

    if out_type == "np":
        x = np.array(x)
    return x

def get_time_embedding(timestep, dtype=torch.float16):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160) / 160)
    # Shape: (1, 160)
    if len(timestep.shape) == 0:
        timestep = torch.tensor([timestep])
    x = timestep.clone().detach()[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1).to(dtype) 

def get_model_weights_dtypes(models_wrapped_dict, verbose=False):
    
    model_dtype_map = {}
    for name, model in models_wrapped_dict.items():
        if isinstance(model, list):
            model = model[0]
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

def extract_euclidien_similarity(data_arr):
    data_arr = np.array(data_arr)
    norm_data = np.sum(data_arr ** 2, axis=1).reshape(-1, 1)
    squared_distances = norm_data + norm_data.T - 2 * np.dot(data_arr, data_arr.T)
    squared_distances = np.maximum(squared_distances, 0)
    distances = np.sqrt(squared_distances)
    similarities = 1 / (1 + distances)
    np.fill_diagonal(similarities, 1)
    
    return similarities

def get_colors_and_ids(palette, color_list):
    palette = np.array(palette)
    color_array = np.array(color_list)+1

    rgb_results = []
    id_results = []
    for target in palette:
        similarities = extract_euclidien_similarity(np.concatenate([target[None,:], color_array]))[0][1:]

        id_results.append(np.argmax(similarities))
        rgb_results.append((color_array[np.argmax(similarities)]-1).tolist())
        
    return rgb_results, id_results

def load_colot_list_data(path="./data/list_of_colors.jsonl"):
    color_list = {}
    with open(path, mode="r") as file:
        for line in file:
            line = json.loads(line)
            color_list[line["color_number"]] = line["color_rgb"]
    color_list = [cur[1] for cur in sorted(color_list.items(), key=lambda x:x[0])]
    return color_list

def composing_image(img1, img2, mask, out_type="pil"):
    img1 = np.array(img1)
    mask = np.array(mask)
    img2 = np.array(img2)
    
    composed_output = img1 * (1-rescale(mask, (np.min(mask), np.max(mask)), (0,1), out_type="np")) + \
        img2 * rescale(mask, (np.min(mask), np.max(mask)), (0,1), out_type="np")
    
    if out_type == "pil":
        composed_output = Image.fromarray(composed_output.astype(np.uint8))
    
    return composed_output
