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

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Favorfit diffusion controlnet train argements")
    parser.add_argument(
        "--diffusion_model_path",
        type=str,
        default="/home/mlfavorfit/lib/favorfit/kjg/0_model_weights/diffusion/v1-5-pruned-emaonly.ckpt",
    )
    parser.add_argument(
        "--clip_image_encoder_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--unet",
        action="store_true",
    )
    parser.add_argument(
        "--clip",
        action="store_true",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--validation_image_paths",
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
        default="no",
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

from networks.clip.clip_image_encoder import CLIPImagePreprocessor
def make_train_dataset(path, accelerator):
    dataset = load_dataset(path)
    column_names = dataset['train'].column_names
    image_column, reference_column = column_names

    image_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    reference_transforms = CLIPImagePreprocessor()

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in images]

        ref_images = [image.convert("RGB") for image in examples[reference_column]]
        ref_images = [reference_transforms(image)[0] for image in ref_images]

        examples["pixel_values"] = images
        examples["ref_values"] = ref_images

        return examples
    
    with accelerator.main_process_first():
        train_dataset = dataset["train"].with_transform(preprocess_train)
    
    return train_dataset

def load_models(args):
    if args.precision == "fp16":
        precison = torch.float16
    else:
        precison = torch.float32

    from utils.model_loader import load_diffusion_model
    if args.diffusion_model_path is not None:
        diffusion_state_dict = torch.load(args.diffusion_model_path)

        if "diffusion" not in diffusion_state_dict.keys():
            from utils.model_converter import convert_model
            diffusion_state_dict = convert_model(diffusion_state_dict)
            
    models = load_diffusion_model(diffusion_state_dict, dtype=precison, **{"unet_dtype":torch.float32 if args.unet == True else None,
                                                                            "clip_train":True if args.clip == True else None,
                                                                            "clip_image_encoder":True,
                                                                            "clip_image_encoder_from_pretrained":True,
                                                                            "clip_image_encoder_model_path":args.clip_image_encoder_model_path,
                                                                            "clip_dtype":torch.float32 if args.clip == True else precison,})

    return models

import wandb
from pipelines.pipeline_reference import generate
from PIL import Image
def log_validation(encoder, decoder, clip, diffusion, accelerator, args):

    models = {}
    models['clip'] = clip
    models['encoder'] = encoder
    models['decoder'] = decoder
    models['diffusion'] = diffusion

    image_logs = []
    for validation_image_path in args.validation_image_paths:
        val_image = Image.open(validation_image_path).convert("RGB")

        output_images = generate(
                ref_image=val_image,
                unref_image=None,
                input_image=None,
                control_image=None,
                num_per_image=3,
                strength=0.8,
                do_cfg=False,
                cfg_scale=7.5,
                sampler_name="ddpm",
                n_inference_steps=20,
                models=models,
                seeds=[42, 110, 320],
                device=accelerator.device,
                idle_device="cuda",
                leave_tqdm=False
            )
            
        images = output_images

        for image in images:
            image_logs.append(
                {"images": image, "validation_image": val_image}
            )


    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                image = log["images"]
                validation_image = log["validation_image"]
                formatted_images.append(wandb.Image(validation_image, caption="reference"))
                formatted_images.append(wandb.Image(image, caption="generated_image"))                

            tracker.log({"validation": formatted_images})

    return image_logs

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    ref_values = torch.stack([example["ref_values"] for example in examples])
    ref_values = ref_values.to(memory_format=torch.contiguous_format).float()

    return {
        "pixel_values": pixel_values,
        "ref_values": ref_values,
    }


import torch.nn.functional as F
from pipelines.utils import get_time_embedding
def train(accelerator,
        train_dataloader,
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
            latents = encoder(batch["pixel_values"].to(dtype=weight_dtype))

            noise = torch.randn_like(latents)
            batch_size = batch['pixel_values'].shape[0]
            
            timesteps = torch.randint(0, sampler.num_train_timesteps, (batch_size,), device="cpu").long()
            
            latents = sampler.add_noise(latents, timesteps, noise)

            contexts = clip(batch['ref_values'].to(next(clip.parameters()).dtype)).to(dtype=weight_dtype)
            
            time_embeddings = get_time_embedding(timesteps).to(latents.device)

            model_pred = diffusion(
                latents,
                contexts,
                time_embeddings
            )

            target = noise
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            accelerator.backward(loss)
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

                        if args.clip == True:
                            clip = accelerator.unwrap_model(clip)
                            torch.save(clip, os.path.join(save_path, f"clip_image_encoder_{epoch}.pth"))
                        if args.unet == True:
                            diffusion = accelerator.unwrap_model(diffusion)
                            torch.save(diffusion, os.path.join(save_path, f"diffusion_{epoch}.pth"))
                    
                    if global_step % args.validation_step == 0:
                        log_validation(encoder,
                                    decoder,
                                    clip,
                                    diffusion,
                                    accelerator,
                                    args)

            logs = {"loss": loss.detach().item(), "lr": optimizer.param_groups[0]['lr']}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            
            if args.clip == True:
                clip = accelerator.unwrap_model(clip)
                torch.save(clip, f"./training/clip_image_encoder_{epoch}.pth")
            if args.unet == True:
                diffusion = accelerator.unwrap_model(diffusion)
                torch.save(diffusion, f"./training/diffusion_{epoch}.pth")


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

    models = load_models(args)

    clip = models['clip']
    encoder = models['encoder']
    decoder = models['decoder'] 
    diffusion = models['diffusion']

    encoder.requires_grad_(False)
    decoder.requires_grad_(False)

    if args.clip == True:
        clip.requires_grad_(True)
        clip.clip_image_encoder.vision_model.post_layernorm.requires_grad_(False)
    else:
        clip.requires_grad_(False)

    if args.unet == True:
        diffusion.requires_grad_(True)
    else:
        diffusion.requires_grad_(False)

    train_dataset = make_train_dataset(args.train_data_path, accelerator)

    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=0
    )
 
    from torch.optim import AdamW
    
    params_to_optimize = []
    if args.clip == True:
        params_to_optimize += list(clip.parameters())
    if args.unet == True:
        params_to_optimize += list(diffusion.parameters())
    
    if args.use_lr_scheduler:
        optimizer = AdamW(
            params_to_optimize,
            lr=3e-06,
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
    
    to_train_models = []
    if args.clip == True:
        to_train_models.append(clip)
    if args.unet == True:
        to_train_models.append(diffusion)

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
        tracker_config.pop("validation_image_paths")

        accelerator.init_trackers("train_reference", config=tracker_config)
    
    encoder.to(accelerator.device, dtype=weight_dtype)
    decoder.to(accelerator.device, dtype=weight_dtype)
    if args.clip != True:
        clip.to(accelerator.device, dtype=weight_dtype)
    if args.unet != True:
        diffusion.to(accelerator.device, dtype=weight_dtype)

    train(accelerator,
        train_dataloader,
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
