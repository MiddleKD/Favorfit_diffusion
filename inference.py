import os
from typing import Union, List, Dict

from transformers import CLIPTokenizer
from pipelines import pipeline_default, pipline_default_controlnet, pipeline_inpainting_controlnet, pipline_positive_controlnet
from .utils.model_loader import *


# Call model
def call_diffusion_model(
        diffusion_state_dict_path: str,
        lora_state_dict_path: Union[str, None] = None,
)-> Dict:
    
    kwargs = {"is_lora": True if lora_state_dict_path is not None else False,
              "is_inpaint": True if "inpaint" in diffusion_state_dict_path else False}
    
    diffusion_state_dict = torch.load(diffusion_state_dict_path)

    if lora_state_dict_path is not None:
        lora_state_dict = torch.load(lora_state_dict_path)
        diffusion_state_dict["lora"] = lora_state_dict

    models = load_diffusion_model(diffusion_state_dict, **kwargs)

    return models

def call_tokenizer()-> CLIPTokenizer:
    
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    tokenizer = CLIPTokenizer(vocab_file=os.path.join(cur_dir, "data/vocab.json"), 
                            merges_file=os.path.join(cur_dir, "data/merges.txt"))

    return tokenizer

def call_controlnet_model(
        control_state_dict_path: Union[str, List[str], None] = None,
)-> Dict:
    
    control_state_dict_list = []
    if not isinstance(control_state_dict_path, list):
        control_state_dict_path = [control_state_dict_path]
    for cur in control_state_dict_path:
        control_state_dict_list.append(torch.load(cur))
    
    controlnet = load_controlnet_model(control_state_dict_list)

    return controlnet

def make_multi_controlnet_model(
        controlnet_model_list
)-> Dict:
    
    outputs_controlnet_model_dict={"controlnet":[], "controlnet_embedding":[]}
    for controlnet_model_dict in controlnet_model_list:
        outputs_controlnet_model_dict["controlnet"].append(*controlnet_model_dict["controlnet"])
        outputs_controlnet_model_dict["controlnet_embedding"].append(*controlnet_model_dict["controlnet_embedding"])
    
    return outputs_controlnet_model_dict


# inference pipeline
default_uncond_prompt = "low quality, worst quality, wrinkled, deformed, distorted, jpeg artifacts,nsfw, paintings, sketches, text, watermark, username, spikey"

def text_to_image(
        prompt,
        uncond_prompt=default_uncond_prompt,
        num_per_image=1,
        lora_scale=0.7,
        models=None,
        seeds=[],
        device="cpu",
        tokenizer=None,
        ):

    output_images = pipeline_default.generate(
        prompt=f"professional photography, natural, {prompt}, realistic, high resolution, 8k",
        uncond_prompt=uncond_prompt,
        input_image=None,
        num_per_image=num_per_image,
        do_cfg=True,
        cfg_scale=7.5,
        sampler_name="ddpm",
        n_inference_steps=20,
        seeds=seeds,
        models=models,
        device=device,
        idle_device="cpu",
        tokenizer=tokenizer,
        lora_scale=lora_scale,
    )

    return output_images


def image_to_image(
        input_image,
        prompt,
        uncond_prompt=default_uncond_prompt,
        num_per_image=1,
        lora_scale=0.7,
        models=None,
        seeds=[], 
        device="cpu",
        tokenizer=None,
        ):

    output_images = pipeline_default.generate(
        prompt=f"professional photography, natural, {prompt}, realistic, high resolution, 8k",
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        num_per_image=num_per_image,
        do_cfg=True,
        cfg_scale=7.5,
        sampler_name="ddpm",
        n_inference_steps=20,
        seeds=seeds,
        models=models,
        device=device,
        idle_device="cpu",
        tokenizer=tokenizer,
        lora_scale=lora_scale,
    )

    return output_images


def text_to_image_controlnet(
        control_image,
        prompt,
        uncond_prompt=default_uncond_prompt,
        num_per_image=1,
        strength=0.8,
        lora_scale=0.7,
        controlnet_scale=1.0,
        models=None,
        seeds=[], 
        device="cpu",
        tokenizer=None,
        ):

    output_images = pipline_default_controlnet.generate(
        prompt=f"professional photography, natural, {prompt}, realistic, high resolution, 8k",
        uncond_prompt=uncond_prompt,
        input_image=None,
        control_image=control_image,
        num_per_image=num_per_image,
        do_cfg=True,
        cfg_scale=7.5,
        sampler_name="ddpm",
        n_inference_steps=20,
        strength=strength,
        models=models,
        seeds=seeds,
        device=device,
        idle_device="cpu",
        tokenizer=tokenizer,
        lora_scale=lora_scale,
        controlnet_scale=controlnet_scale
    )

    return output_images


def image_to_image_controlnet(
        input_image,
        control_image,
        prompt,
        uncond_prompt=default_uncond_prompt,
        num_per_image=1,
        strength=0.8,
        lora_scale=0.7,
        controlnet_scale=1.0,
        models=None,
        seeds=[], 
        device="cpu",
        tokenizer=None,
        ):

    output_images = pipline_default_controlnet.generate(
        prompt=f"professional photography, natural, {prompt}, realistic, high resolution, 8k",
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        control_image=control_image,
        num_per_image=num_per_image,
        do_cfg=True,
        cfg_scale=7.5,
        sampler_name="ddpm",
        n_inference_steps=20,
        strength=strength,
        models=models,
        seeds=seeds,
        device=device,
        idle_device="cpu",
        tokenizer=tokenizer,
        lora_scale=lora_scale,
        controlnet_scale=controlnet_scale
    )

    return output_images


def inpainting_controlnet(
        input_image,
        mask_image,
        control_image,
        prompt,
        uncond_prompt=default_uncond_prompt,
        num_per_image=1,
        strength=0.6,
        lora_scale=0.7,
        controlnet_scale=1.0,
        models=None,
        seeds=[], 
        device="cpu",
        tokenizer=None,
        ):

    output_images = pipeline_inpainting_controlnet.generate(
        prompt=f"professional photography, natural, {prompt}, realistic, high resolution, 8k",
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        mask_image=mask_image,
        control_image=control_image,
        num_per_image=num_per_image,
        do_cfg=True,
        cfg_scale=7.5,
        sampler_name="ddpm",
        n_inference_steps=20,
        strength=strength,
        models=models,
        seeds=seeds,
        device=device,
        idle_device="cpu",
        tokenizer=tokenizer,
        lora_scale=lora_scale,
        controlnet_scale=controlnet_scale
    )

    return output_images

def text_to_image_positive_controlnet(
        control_image,
        positive_control_image,
        prompt,
        uncond_prompt=default_uncond_prompt,
        num_per_image=1,
        lora_scale=0.7,
        controlnet_scale=[1.0, 1.0],
        models=None,
        seeds=[], 
        device="cpu",
        tokenizer=None,
        ):

    output_images = pipline_positive_controlnet.generate(
        prompt=f"professional photography, natural, {prompt}, realistic, high resolution, 8k",
        uncond_prompt=uncond_prompt,
        input_image=None,
        control_image=control_image,
        positive_control_image=positive_control_image,
        num_per_image=num_per_image,
        do_cfg=True,
        cfg_scale=7.5,
        sampler_name="ddpm",
        n_inference_steps=20,
        models=models,
        seeds=seeds,
        device=device,
        idle_device="cpu",
        tokenizer=tokenizer,
        lora_scale=lora_scale,
        controlnet_scale=controlnet_scale
    )

    return output_images
