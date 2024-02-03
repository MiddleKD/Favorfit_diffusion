import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from models.scheduler.ddpm import DDPMSampler
from pipelines.utils import rescale, get_time_embedding, get_model_weights_dtypes, prepare_latent_width_height


def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    mask_image=None,
    num_per_image=1,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seeds=None,
    device=None,
    idle_device=None,
    tokenizer=None,
    leave_tqdm=True,
    **kwargs
):
    ORIGIN_WIDTH, ORIGIN_HEIGHTS, WIDTH, HEIGHT, LATENTS_WIDTH, LATENTS_HEIGHT = prepare_latent_width_height([input_image, mask_image])

    with torch.no_grad():
        dtype_map = get_model_weights_dtypes(models_wrapped_dict=models)

        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # For batch-----
        generators = [torch.Generator(device=device) for _ in range(num_per_image)]
        
        if isinstance(seeds,int):
            seeds = [seeds]*num_per_image
        elif len(seeds) != num_per_image: 
            print("length do not match. seeds set to None.")
            seeds = None
        
        if seeds is None:
            for generator in generators:
                generator.seed()
        else:
            for generator, seed in zip(generators, seeds):
                generator.manual_seed(seed)
        # For batch-----

        clip = models["clip"]
        clip.to(device)
        
        if do_cfg:
            # Convert into a list of length Seq_Len=77
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            cond_context = clip(cond_tokens)
            # Convert into a list of length Seq_Len=77
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            uncond_context = clip(uncond_tokens)
            # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
            context = torch.cat([cond_context, uncond_context])

            # For batch-----
            context = context.repeat_interleave(num_per_image, dim=0)
            # For batch-----

        else:
            # Convert into a list of length Seq_Len=77
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            context = clip(tokens)

            # For batch-----
            context = context.repeat_interleave(num_per_image, dim=0)
            # For batch-----
        to_idle(clip)

        if sampler_name == "ddpm":
            # For batch-----
            samplers = []
            for generator in generators:
                sampler = DDPMSampler(generator)
                sampler.set_inference_timesteps(n_inference_steps)
                samplers.append(sampler)
            # For batch-----
        else:
            raise ValueError("Unknown sampler value %s. ")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            # (Height, Width, Channel)
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2).to(device)

            # For batch-----
            input_image_tensor = input_image_tensor.repeat_interleave(num_per_image, dim=0)
            
            encoder_noise = torch.cat([torch.randn(latents_shape, 
                                                   generator=generator, 
                                                   device=device) 
                                                   for generator in generators
                                    ])

            input_image_tensor = input_image_tensor.to(dtype=dtype_map["encoder"])
            encoder_noise = encoder_noise.to(dtype=dtype_map["encoder"])

            latents = []
            for idx, (its, en) in enumerate(zip(input_image_tensor, encoder_noise)):
                sampler = samplers[idx]
                sampler.set_strength(strength=strength)
                latent = encoder(its.unsqueeze(0), en.unsqueeze(0))
                latent = sampler.add_noise(latent, sampler.timesteps[0])
                latents.append(latent)
            latents = torch.cat(latents)
            # For batch-----

            input_image_np = input_image.resize((WIDTH, HEIGHT))
            input_image_np = np.array(input_image_np)
            
            mask_from_pil = torch.from_numpy(np.array(mask_image) / 255.0)
            mask_from_pil = mask_from_pil.unsqueeze(-1).unsqueeze(0).permute(0, 3, 1, 2)
            mask = torch.nn.functional.interpolate(mask_from_pil, size=(WIDTH//8, HEIGHT//8))
            mask = mask.to(device)
            
            # For batch-----
            mask = mask.repeat_interleave(num_per_image, dim=0)
            # For batch-----
            
            mask_condition = np.array(mask_image.resize((WIDTH, HEIGHT)))
            masked_image_tensor = rescale(torch.tensor(input_image_np), (0, 255), (-1, 1)).numpy() * (mask_condition[:,:,None] < 100)

            masked_image_tensor = torch.tensor(masked_image_tensor)
            # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            masked_image_tensor = masked_image_tensor.unsqueeze(0)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            masked_image_tensor = masked_image_tensor.permute(0, 3, 1, 2).to(device)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            masked_image_latents = encoder(masked_image_tensor, encoder_noise)

            to_idle(encoder)
        else:
            # For batch-----
            latents = torch.cat([torch.randn(latents_shape, 
                                            generator=generator, 
                                            device=device) 
                                            for generator in generators]
                                )
            # For batch-----
        
        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps, leave=leave_tqdm)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents

            if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = torch.cat([model_input.repeat(2, 1, 1, 1), 
                                         mask.repeat(2, 1, 1, 1), 
                                         masked_image_latents.repeat(2, 1, 1, 1)], 
                                         dim=1)
            else:
                model_input = torch.cat([model_input, 
                                         mask, 
                                         masked_image_latents], 
                                         dim=1)
            
            # model_output is the predicted noise
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            model_output = diffusion(model_input.to(dtype=dtype_map["diffusion"]), 
                                     context.to(dtype=dtype_map["diffusion"]), 
                                     time_embedding.to(dtype=dtype_map["diffusion"]),
                                     **kwargs)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        images = decoder(latents.to(dtype=dtype_map["decoder"]))
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        
        return [Image.fromarray(image).resize([ORIGIN_WIDTH, ORIGIN_HEIGHTS]) for image in images]

