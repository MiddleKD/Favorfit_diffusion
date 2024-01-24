import torch
import numpy as np
from tqdm import tqdm
from models.scheduler.ddpm import DDPMSampler
from pipelines.utils import rescale, get_time_embedding, get_model_weights_dtypes

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    color_palette=None,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    strength=0.8,
    models={},
    seed=None,
    device=None,
    idle_device="cpu",
    tokenizer=None,
    leave_tqdm=True
):
    with torch.no_grad():
        dtype_map = get_model_weights_dtypes(models_wrapped_dict=models)

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x


        # Initialize random number generator according to the seed specified
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)
        color_palette_embedding_model = models["color_palette_embedding"]
        color_palette_embedding_model.to(device)
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

            # This is Color Palette Embedding Model---------------------------
            color_palette = torch.tensor(color_palette).unsqueeze(0).to(device)
            color_palette_embedding = color_palette_embedding_model(color_palette.to(dtype=dtype_map["color_palette_embedding"]))

            uncon_color_palette = torch.zeros_like(color_palette).to(device)
            uncon_color_palette_embedding = color_palette_embedding_model(uncon_color_palette.to(dtype=dtype_map["color_palette_embedding"]))

            cond_context = torch.cat([cond_context, color_palette_embedding], 1)
            uncond_context = torch.cat([uncond_context, uncon_color_palette_embedding], 1)
            #-------------------------------------------------------------------

            # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Convert into a list of length Seq_Len=77
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            context = clip(tokens)

            # This is Color Palette Embedding Model---------------------------
            color_palette = torch.tensor(color_palette).to(device)
            color_palette_embedding = color_palette_embedding_model(color_palette.to(dtype=dtype_map["color_palette_embedding"]))

            context = torch.cat([context, color_palette_embedding], 1)

        to_idle(clip)
        to_idle(color_palette_embedding_model)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
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

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = encoder(input_image_tensor.to(dtype=dtype_map["encoder"]), 
                              encoder_noise.to(dtype=dtype_map["encoder"]))

            # Add noise to the latents (the encoded input image)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = torch.randn(latents_shape, generator=generator, device=device)


        # diffusion
        diffusion = models["diffusion"]
        diffusion.to(device)
        color_palette_timestep_embedding_model = models["color_palette_timestep_embedding"]
        color_palette_timestep_embedding_model.to(device)
        
        timesteps = tqdm(sampler.timesteps, leave=leave_tqdm)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep, dtype=dtype_map["color_palette_timestep_embedding"]).to(device)
            color_palette_timestep_embedding = color_palette_timestep_embedding_model(color_palette.to(dtype=dtype_map["color_palette_timestep_embedding"]))
            time_embedding += color_palette_timestep_embedding

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents

            if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            model_output = diffusion(model_input.to(dtype=dtype_map["diffusion"]), 
                                    context.to(dtype=dtype_map["diffusion"]), 
                                    time_embedding.to(dtype=dtype_map["diffusion"]),
            )

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)
        to_idle(color_palette_timestep_embedding_model)


        decoder = models["decoder"]
        decoder.to(device)
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        images = decoder(latents.to(dtype=dtype_map["decoder"]))
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()

        return images[0]
