if __name__ == "__main__":
    import sys, os
    sys.path.append(os.getcwd())

import os
import argparse
from PIL import Image
import torch
from transformers import CLIPTokenizer
from utils import model_loader
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="diffusion inference test code")
    parser.add_argument(
        "--root_path",
        type=str,
        default="/mnt/sdb/0_model_weights/diffusion",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="diffusion_test",
    )
    args = parser.parse_args()
    return args


DEVICE = "cpu"
tokenizer = None


def initalize(args):
    global DEVICE, tokenizer

    torch.cuda.empty_cache()

    DEVICE = "cpu"

    ALLOW_CUDA = True
    ALLOW_MPS = False

    if torch.cuda.is_available() and ALLOW_CUDA:
        DEVICE = "cuda"
    elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
        DEVICE = "mps"

    print(f"Using device: {DEVICE}")

    tokenizer = CLIPTokenizer("./data/vocab.json", merges_file="./data/merges.txt")
    print("tokenizer call OK")


def wandb_init(args):
    wandb.login()
    wandb.init(project=args.project_name)


def text_to_image(args):
    torch.cuda.empty_cache()

    state_dict = torch.load(os.path.join(args.root_path,"favorfit_base.pth"))
    models = model_loader.load_diffusion_model(state_dict)

    from pipelines.pipeline_default import generate
    output_image = generate(
        prompt="A blue bottle",
        uncond_prompt="deform, low quality",
        input_image=None,
        do_cfg=True,
        cfg_scale=8,
        sampler_name="ddpm",
        n_inference_steps=20,
        seed=42,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )

    output_image_pil = Image.fromarray(output_image)
    print("Test: text_to_image OK")
    return output_image_pil


def inpainting(args):
    torch.cuda.empty_cache()

    def make_inpaint_data(image_path, mask_path):
        img_pil = Image.open(image_path).convert("RGB")
        mask_pil = Image.open(mask_path).convert("L")
        return img_pil, mask_pil
    
    state_dict = torch.load(os.path.join(args.root_path,"favorfit_inpaint.pth"))
    kwargs = {"is_inpaint":True}
    models = model_loader.load_diffusion_model(state_dict, **kwargs)

    image_path = "./images/test_image/bottle.jpg"
    mask_path = "./images/test_image/mask.png"
    input_image, mask = make_inpaint_data(image_path, mask_path)

    from pipelines.pipeline_inpainting import generate
    output_image = generate(
        prompt="red bottle cap, high resolution, 8k",
        uncond_prompt="deform, low quality",
        input_image=input_image,
        mask_image=mask,
        strength=1.0,
        do_cfg=True,
        cfg_scale=8,
        sampler_name="ddpm",
        n_inference_steps=20,
        seed=42,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )

    output_image_pil = Image.fromarray(output_image)
    print("Test: inpainting OK")
    return output_image_pil


def controlnet(args):
    torch.cuda.empty_cache()

    diffusion_state_dict = torch.load(os.path.join(args.root_path,"favorfit_base.pth"))
    control_state_dict = torch.load(os.path.join(args.root_path,"controlnet","outpaint_v2.pth"))   

    kwargs = {"is_controlnet":True, "controlnet_scale":1.0}
    models = model_loader.load_diffusion_model(diffusion_state_dict, **kwargs)
    controlnet = model_loader.load_controlnet_model(control_state_dict)

    models.update(controlnet)

    control_image = Image.open("./images/test_image/object_outpaint.jpg")
    control_image = control_image.resize((512,512))

    from pipelines.pipline_controlnet import generate
    output_image = generate(
        prompt="A coffe cup and lemons",
        uncond_prompt="deform, low quality",
        input_image=None,
        control_image=control_image,
        do_cfg=True,
        cfg_scale=7.5,
        sampler_name="ddpm",
        n_inference_steps=20,
        strength=1.0,
        models=models,
        seed=12345,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer
    )

    output_image_pil = Image.fromarray(output_image)
    print("Test: controlnet OK")
    return output_image_pil


def lora(args):
    torch.cuda.empty_cache()

    diffusion_state_dict = torch.load(os.path.join(args.root_path,"favorfit_base.pth"))
    lora_state_dict = torch.load(os.path.join(args.root_path,"lora","favorfit_lora.pth"))   
    diffusion_state_dict["lora"] = lora_state_dict

    kwargs = {"is_lora":True, "lora_scale":1.0}
    models = model_loader.load_diffusion_model(diffusion_state_dict, **kwargs)

    from pipelines.pipeline_default import generate
    output_image = generate(
        prompt="A blue bottle",
        uncond_prompt="deform, low quality",
        input_image=None,
        do_cfg=True,
        cfg_scale=8,
        sampler_name="ddpm",
        n_inference_steps=20,
        seed=42,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )

    output_image_pil = Image.fromarray(output_image)
    print("Test: lora OK")
    return output_image_pil

def main(args):
    initalize(args)
    wandb_init(args)
    
    text_to_image_result = text_to_image(args)
    inpainting_result = inpainting(args)
    controlnet_result = controlnet(args)
    lora_result = lora(args)

    wandb.log({"text_to_image": wandb.Image(text_to_image_result), 
               "inpainting":wandb.Image(inpainting_result),
               "controlnet":wandb.Image(controlnet_result),
               "lora":wandb.Image(lora_result)})
    
    print("All tests OK")

if __name__ == "__main__":
    args = parse_args()
    main(args)