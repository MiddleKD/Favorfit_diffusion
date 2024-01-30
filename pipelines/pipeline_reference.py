import torch
import numpy as np
from tqdm import tqdm
from models.scheduler.ddpm import DDPMSampler
from models.clip.clip_image_encoder import CLIPImagePreprocessor
from pipelines.utils import rescale, get_time_embedding, get_model_weights_dtypes

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
    ref_image,
    unref_image=None,
    input_image=None,
    control_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    leave_tqdm=True
):
    with torch.no_grad():
        dtype_map = get_model_weights_dtypes(models_wrapped_dict=models)

        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

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
        
        image_preprocessor = CLIPImagePreprocessor()
        if do_cfg:
            cond_context = clip(image_preprocessor(ref_image))
            uncond_context = clip(image_preprocessor(unref_image))
            context = torch.cat([cond_context, uncond_context])
        else:
            cond_context = clip(image_preprocessor(ref_image))
        to_idle(clip)

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


        # controlnet embedding
        if control_image is not None:
            control_image_tensor = control_image.resize((WIDTH, HEIGHT))
            control_image_tensor = np.array(control_image_tensor)
            control_image_tensor = torch.tensor(control_image_tensor)
            control_image_tensor = rescale(control_image_tensor, (0, 255), (0, 1))
            control_image_tensor = control_image_tensor.unsqueeze(0)
            control_image_tensor = control_image_tensor.permute(0, 3, 1, 2).to(device)
            
            controlnet_embedding_model = models["controlnet_embedding"]
            controlnet_embedding_model.to(device)
            control_embedding_latent = controlnet_embedding_model(control_image_tensor.to(dtype=dtype_map["controlnet_embedding"]))
            to_idle(controlnet_embedding_model)


        # diffusion
        diffusion = models["diffusion"]
        diffusion.to(device)

        if control_image is not None:
            controlnet_model = models["controlnet"]
            controlnet_model.to(device)

        timesteps = tqdm(sampler.timesteps, leave=leave_tqdm)

        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            if control_image is not None:
                # controlnet
                control_img_input = latents
                control_embedding_input = control_embedding_latent

                if do_cfg:
                    # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                    control_img_input = control_img_input.repeat(2, 1, 1, 1)
                
                # model_output is the predicted noise
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
                controlnet_downs, controlnet_mids = controlnet_model(original_sample=control_img_input.to(dtype=dtype_map["controlnet"]), 
                                                                    latent=control_embedding_input.to(dtype=dtype_map["controlnet"]), 
                                                                    context=context.to(dtype=dtype_map["controlnet"]), 
                                                                    time=time_embedding.to(dtype=dtype_map["controlnet"]))
                #---------------------------------------------------------------------
            
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
                                    additional_res_condition=[
                                        [cur.to(dtype=dtype_map["diffusion"]) for cur in controlnet_downs], 
                                        [cur.to(dtype=dtype_map["diffusion"]) for cur in controlnet_mids]
                                    ] if control_image is not None else None
            )

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = sampler.step(timestep, latents, model_output)

        if control_image is not None:
            to_idle(controlnet_model)
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
        return images[0]
