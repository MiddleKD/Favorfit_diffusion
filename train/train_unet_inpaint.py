if __name__ == "__main__":
    import sys, os
    sys.path.append(os.getcwd())

import os
from tqdm import tqdm
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import Accelerator
from datasets import load_dataset
import torch
from torchvision import transforms
from train_utils import update_ckpt_weights

import argparse
import json
def parse_palette_argument(palette_string):
    return json.loads(palette_string)

def parse_args():
    parser = argparse.ArgumentParser(description="Favorfit diffusion inpaint train argements")
    parser.add_argument(
        "--diffusion_model_path",
        type=str,
        default="/home/mlfavorfit/lib/favorfit/kjg/0_model_weights/diffusion/v1-5-pruned-emaonly.ckpt",
    )
    parser.add_argument(
        "--resume_ckpt_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--validation_images",
        type=str,
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--validation_masks",
        type=str,
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--save_ckpt_step",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--validation_step",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        choices=["wandb"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--use_lr_scheduler",
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    args = parser.parse_args()
    return args
import numpy as np
def make_train_dataset(path, tokenizer, accelerator):
    dataset = load_dataset(path)
    column_names = dataset['train'].column_names
    image_column, mask_column, caption_column = column_names

    image_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    mask_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
        ]
    )
    mask_latents_transforms = transforms.Compose(
        [
            transforms.Resize(64, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in images]

        masks_pil_list = [mask.convert("L") for mask in examples[mask_column]]
        # masks_pil_list = [Image.fromarray(255 - np.array(mask.convert("L"))) for mask in examples[mask_column]]
        masks = [mask_transforms(mask) for mask in masks_pil_list]
        masks_latent = [mask_latents_transforms(mask) for mask in masks_pil_list]

        black_image = torch.zeros_like(images[0])
        masked_images = [black_image * (mask) + image * (1-mask) for image, mask in zip(images, masks)]

        tokenized_ids = tokenizer.batch_encode_plus([cur for cur in examples[caption_column]], padding="max_length", max_length=77, truncation=True).input_ids
        
        examples["pixel_values"] = images
        examples["masks"] = masks_latent
        examples["masked_images"] = masked_images
        examples["input_ids"] = tokenized_ids

        return examples
    
    with accelerator.main_process_first():
        train_dataset = dataset["train"].with_transform(preprocess_train)
    
    return train_dataset

def load_models(args):
    if args.precision == "fp16":
        precison = torch.float16
    else:
        precison = torch.float32

    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer("./data/vocab.json", merges_file="./data/merges.txt")

    from utils.model_loader import load_diffusion_model
    if args.diffusion_model_path is not None:
        diffusion_state_dict = torch.load(args.diffusion_model_path)

        if "diffusion" not in diffusion_state_dict.keys():
            from utils.model_converter import convert_model
            diffusion_state_dict = convert_model(diffusion_state_dict)
    
    kwargs = {"is_inpaint":True}
    models = load_diffusion_model(diffusion_state_dict, dtype=precison, **kwargs)

    if args.resume_ckpt_path is not None:
        models = update_ckpt_weights(models, root_ckpt_path=args.resume_ckpt_path)

    return models, tokenizer

import wandb
from pipelines.pipeline_inpainting import generate
from PIL import Image
def log_validation(encoder, decoder, clip, tokenizer, diffusion, accelerator, args):

    models = {}
    models['clip'] = clip
    models['encoder'] = encoder
    models['decoder'] = decoder
    models['diffusion'] = diffusion

    image_logs = []
    for validation_prompt, validation_image, validation_mask in zip(args.validation_prompts, args.validation_images, args.validation_masks):
        validation_image = Image.open(validation_image).convert("RGB")
        validation_mask = Image.open(validation_mask).convert("L")
        # validation_mask = Image.fromarray(255 - np.array(Image.open(validation_mask).convert("L")))

        output_images = generate(
            prompt=validation_prompt,
            uncond_prompt="deform, low quality",
            input_image=validation_image,
            mask_image=validation_mask,
            num_per_image=3,
            do_cfg=True,
            strength=0.6,
            cfg_scale=7.5,
            sampler_name="ddpm",
            n_inference_steps=20,
            seeds=[12345, 42, 110],
            models=models,
            device=accelerator.device,
            idle_device="cuda",
            tokenizer=tokenizer,
            leave_tqdm=False
        )

        images = output_images

        for image in images:
            image_logs.append(
                {"images": image, "validation_prompts": validation_prompt, "validation_images": validation_image, "validation_masks": validation_mask}
            )

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                image = log["images"]
                validation_prompt = log["validation_prompts"]
                validation_image = log["validation_images"]
                validation_mask = log["validation_masks"]
                formatted_images.append(wandb.Image(validation_image, caption=validation_prompt))
                formatted_images.append(wandb.Image(validation_mask, caption="mask"))
                formatted_images.append(wandb.Image(image, caption="result"))

            tracker.log({"validation": formatted_images})

    return image_logs

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    masks = torch.stack([example["masks"] for example in examples])
    masks = masks.to(memory_format=torch.contiguous_format).float()

    masked_images = torch.stack([example["masked_images"] for example in examples])
    masked_images = masked_images.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)

    return {
        "pixel_values": pixel_values,
        "masks": masks,
        "masked_images":masked_images,
        "input_ids": input_ids
    }


import torch.nn.functional as F
from pipelines.utils import get_time_embedding
def train(accelerator,
        train_dataloader,
        tokenizer,
        clip,
        encoder,
        decoder,
        diffusion,
        sampler,
        optimizer,
        lr_scheduler,
        weight_dtype,
        args):
    
    global_step = 0
    progress_bar = tqdm(
        range(0, args.epochs * len(train_dataloader)),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(args.epochs):
        for step, batch in enumerate(train_dataloader):
            image_latents = encoder(batch["pixel_values"].to(dtype=weight_dtype))
            mask_latents = batch["masks"].to(dtype=weight_dtype)
            masked_image_latents = encoder(batch["masked_images"].to(dtype=weight_dtype))
        
            noise = torch.randn_like(image_latents)
            batch_size = batch['pixel_values'].shape[0]
            
            timesteps = torch.randint(0, sampler.num_train_timesteps, (batch_size,), device="cpu").long()
            
            image_latents = sampler.add_noise(image_latents, timesteps, noise)

            latents = torch.cat([image_latents,
                                 mask_latents,
                                 masked_image_latents],
                                 dim=1)
            
            contexts = clip(batch['input_ids']).to(dtype=weight_dtype)

            time_embeddings = get_time_embedding(timesteps).to(latents.device)

            model_pred = diffusion(
                latents,
                contexts,
                time_embeddings
            )

            target = noise
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = ((loss * mask_latents).sum([1, 2, 3]) / mask_latents.sum([1, 2, 3])).mean()

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = list(diffusion.parameters())
                accelerator.clip_grad_norm_(params_to_clip, 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=False)


            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:

                    if global_step % args.save_ckpt_step == 0:
                        save_path = os.path.join("./training", f"checkpoint-{global_step}")
                        os.makedirs(save_path,exist_ok=True)

                        diffusion = accelerator.unwrap_model(diffusion)
                        torch.save(diffusion.state_dict(), os.path.join(save_path, f"diffusion_{epoch}.pth"))
                    
                    if global_step % args.validation_step == 0:
                        log_validation(encoder,
                                    decoder,
                                    clip,
                                    tokenizer,
                                    diffusion,
                                    accelerator,
                                    args)

            logs = {"loss": loss.detach().item(), "lr": optimizer.param_groups[0]['lr']}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:

            diffusion = accelerator.unwrap_model(diffusion)
            torch.save(diffusion.state_dict(), f"./training/diffusion_{epoch}.pth")


def main(args):
    cur_dir = os.path.dirname(os.path.abspath(__name__))
    os.makedirs(os.path.join(cur_dir, "training"), exist_ok=True)
    os.makedirs(os.path.join(cur_dir, "training", "log"), exist_ok=True)

    accelerator_project_config = ProjectConfiguration(
        project_dir=os.path.join(cur_dir, "training"),
        logging_dir=os.path.join(cur_dir, "training", "log")
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=args.precision,
        log_with= args.report_to,
        project_config=accelerator_project_config
    )

    generator = torch.Generator(device=args.device)

    if args.seed is not None:
        set_seed(args.seed)
        generator.manual_seed(args.seed)
    else:
        set_seed(42)
        generator.manual_seed(42)

    models, tokenizer = load_models(args)

    clip = models['clip']
    encoder = models['encoder']
    decoder = models['decoder'] 
    diffusion = models['diffusion']

    encoder.requires_grad_(False)
    decoder.requires_grad_(False)
    clip.requires_grad_(False)
    diffusion.requires_grad_(True)

    train_dataset = make_train_dataset(args.train_data_path, tokenizer, accelerator)

    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=0
    )


    from torch.optim import AdamW
    params_to_optimize = list(diffusion.parameters())


    if args.use_lr_scheduler:
        optimizer = AdamW(
            params_to_optimize,
            lr=1e-06,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-08,
        )
        from networks.lr_scheduler.cosine_base import CosineAnnealingWarmUpRestarts
        lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10000, T_mult=1, eta_max=args.lr,  T_up=20, gamma=1)
    else:
        optimizer = AdamW(
                params_to_optimize,
                lr=args.lr,
                betas=(0.9, 0.999),
                weight_decay=1e-2,
                eps=1e-08,
            )
        from torch.optim.lr_scheduler import LambdaLR
        lr_scheduler = LambdaLR(optimizer, lambda _: 1, last_epoch=-1)

    
    from networks.scheduler.ddpm import DDPMSampler
    sampler = DDPMSampler(generator)
    
    to_train_models = [diffusion]

    *to_train_models, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        *to_train_models, optimizer, train_dataloader, lr_scheduler
    )
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        tracker_config.pop("validation_images")
        tracker_config.pop("validation_masks")

        accelerator.init_trackers("train_unet_inpaint", config=tracker_config)
    
    clip.to(accelerator.device, dtype=weight_dtype)
    encoder.to(accelerator.device, dtype=weight_dtype)
    decoder.to(accelerator.device, dtype=weight_dtype)
    diffusion.to(accelerator.device, dtype=torch.float32)

    train(accelerator,
        train_dataloader,
        tokenizer,
        clip,
        encoder,
        decoder,
        diffusion,
        sampler,
        optimizer,
        lr_scheduler,
        weight_dtype,
        args)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
