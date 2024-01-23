# Script for converting a HF Diffusers saved pipeline to a ControlNet checkpoint.
# *Only* converts the ControlNet.
# Does not convert optimizer state or any other thing.

import argparse
import os.path as osp
import re

import torch
from safetensors.torch import load_file, save_file


# =================#
# UNet Conversion #
# =================#

unet_conversion_map = [
    # (stable-diffusion, HF Diffusers)
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("middle_block_out.0.weight", "controlnet_mid_block.weight"),
    ("middle_block_out.0.bias", "controlnet_mid_block.bias"),
]

unet_conversion_map_resnet = [
    # (stable-diffusion, HF Diffusers)
    ("in_layers.0", "norm1"),
    ("in_layers.2", "conv1"),
    ("out_layers.0", "norm2"),
    ("out_layers.3", "conv2"),
    ("emb_layers.1", "time_emb_proj"),
    ("skip_connection", "conv_shortcut"),
]

unet_conversion_map_layer = []
# hardcoded number of downblocks and resnets/attentions...
# would need smarter logic for other networks.
for i in range(4):
    # loop over downblocks/upblocks

    for j in range(2):
        # loop over resnets/attentions for downblocks
        hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
        sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
        unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

        if i < 3:
            # no attention layers in down_blocks.3
            hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
            sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
            unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

    if i < 3:
        # no downsample in down_blocks.3
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
        sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
        unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

hf_mid_atn_prefix = "mid_block.attentions.0."
sd_mid_atn_prefix = "middle_block.1."
unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

for j in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{j}."
    sd_mid_res_prefix = f"middle_block.{2*j}."
    unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

# controlnet specific

controlnet_cond_embedding_names = ['conv_in'] + [f'blocks.{i}' for i in range(6)] + ['conv_out']
for i, hf_prefix in enumerate(controlnet_cond_embedding_names):
    hf_prefix = f"controlnet_cond_embedding.{hf_prefix}."
    sd_prefix = f"input_hint_block.{i*2}."
    unet_conversion_map_layer.append((sd_prefix, hf_prefix))

for i in range(12):
    hf_prefix = f"controlnet_down_blocks.{i}."
    sd_prefix = f"zero_convs.{i}.0."
    unet_conversion_map_layer.append((sd_prefix, hf_prefix))

def convert_unet_state_dict(unet_state_dict):
    # buyer beware: this is a *brittle* function,
    # and correct output requires that all of these pieces interact in
    # the exact order in which I have arranged them.
    mapping = {k: k for k in unet_state_dict.keys()}
    for sd_name, hf_name in unet_conversion_map:
        mapping[hf_name] = sd_name
    for k, v in mapping.items():
        if "resnets" in k:
            for sd_part, hf_part in unet_conversion_map_resnet:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    for k, v in mapping.items():
        for sd_part, hf_part in unet_conversion_map_layer:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    new_state_dict = {"control_model."+v: unet_state_dict[k] for k, v in mapping.items()}
    return new_state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default=None, type=str, required=True, help="Path to the model to convert.")
    parser.add_argument("--checkpoint_path", default=None, type=str, required=True, help="Path to the output model.")
    parser.add_argument("--half", action="store_true", help="Save weights in half precision.")
    parser.add_argument(
        "--use_safetensors", action="store_true", help="Save weights use safetensors, default is ckpt."
    )

    args = parser.parse_args()

    assert args.model_path is not None, "Must provide a model path!"

    assert args.checkpoint_path is not None, "Must provide a checkpoint path!"

    # Path for safetensors
    # unet_path = osp.join(args.model_path, "diffusion_pytorch_model.safetensors")
    unet_path = args.model_path

    # Load models from safetensors if it exists, if it doesn't pytorch
    if osp.exists(unet_path):
        unet_state_dict = load_file(unet_path, device="cpu")
    else:
        unet_path = osp.join(args.model_path, "diffusion_pytorch_model.bin")
        unet_state_dict = torch.load(unet_path, map_location="cpu")

    # Convert the UNet model
    unet_state_dict = convert_unet_state_dict(unet_state_dict)
    # unet_state_dict = {"model.diffusion_model." + k: v for k, v in unet_state_dict.items()}

    # Put together new checkpoint
    state_dict = {**unet_state_dict, }
    if args.half:
        state_dict = {k: v.half() for k, v in state_dict.items()}

    if args.use_safetensors:
        save_file(state_dict, args.checkpoint_path)
    else:
        state_dict = {"state_dict": state_dict}
        torch.save(state_dict, args.checkpoint_path)