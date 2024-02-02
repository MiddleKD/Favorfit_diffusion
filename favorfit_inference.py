import os
from PIL import Image

from transformers import CLIPTokenizer
from pipelines import pipeline_default, pipline_default_controlnet

cur_dir = os.path.dirname(os.path.realpath(__file__))
tokenizer = CLIPTokenizer(vocab_file=os.path.join(cur_dir, "data/vocab.json"), 
                          merges_file=os.path.join(cur_dir, "data/merges.txt"))
default_uncond_prompt = "low quality, worst quality, wrinkled, deformed, distorted, jpeg artifacts,nsfw, paintings, sketches, text, watermark, username, spikey"


def text_to_image(
        prompt,
        uncond_prompt=default_uncond_prompt,
        num_per_image=1,
        lora_scale=0.7,
        models=None,
        seeds=[], 
        device="cpu"
        ):

    output_images = pipeline_default.generate(
        prompt=f"professional photography, natural shadow, {prompt}, realistic, high resolution, 8k",
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

    output_images_pil = [Image.fromarray(output_image) for output_image in output_images]
    return output_images_pil


def image_to_image(
        input_image,
        prompt,
        uncond_prompt=default_uncond_prompt,
        num_per_image=1,
        lora_scale=0.7,
        models=None,
        seeds=[], 
        device="cpu"
        ):

    output_images = pipeline_default.generate(
        prompt=f"professional photography, natural shadow, {prompt}, realistic, high resolution, 8k",
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

    output_images_pil = [Image.fromarray(output_image) for output_image in output_images]
    return output_images_pil


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
        device="cpu"
        ):

    output_images = pipline_default_controlnet.generate(
        prompt=f"professional photography, natural shadow, {prompt}, realistic, high resolution, 8k",
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

    output_images_pil = [Image.fromarray(output_image) for output_image in output_images]
    return output_images_pil


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
        device="cpu"
        ):

    output_images = pipline_default_controlnet.generate(
        prompt=f"professional photography, natural shadow, {prompt}, realistic, high resolution, 8k",
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

    output_images_pil = [Image.fromarray(output_image) for output_image in output_images]
    return output_images_pil


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
        device="cpu"
        ):

    output_images = pipline_default_controlnet.generate(
        prompt=f"professional photography, natural shadow, {prompt}, realistic, high resolution, 8k",
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

    output_images_pil = [Image.fromarray(output_image) for output_image in output_images]
    return output_images_pil
