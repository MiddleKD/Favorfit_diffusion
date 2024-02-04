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
    control_image=None,
    num_per_image=1,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    strength=0.8,
    models={},
    seeds=None,
    device=None,
    idle_device="cpu",
    tokenizer=None,
    leave_tqdm=True,
    controlnet_scale=1.0,
    **kwargs
):  
    # make latent shape base on input image
    ORIGIN_WIDTH, ORIGIN_HEIGHTS, WIDTH, HEIGHT, LATENTS_WIDTH, LATENTS_HEIGHT = prepare_latent_width_height([input_image, control_image])
    latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

    with torch.no_grad():

        # prepare model type
        dtype_map = get_model_weights_dtypes(models_wrapped_dict=models)

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x


        # check seeds
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


        # prepare scheduler 
        if sampler_name == "ddpm":
            samplers = []
            for generator in generators:
                sampler = DDPMSampler(generator)
                sampler.set_inference_timesteps(n_inference_steps)
                samplers.append(sampler)
        else:
            raise ValueError("Unknown sampler value %s. ")
        

        # 1. CLIP forward
        clip = models["clip"]
        clip.to(device)
        
        if do_cfg:
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            cond_context = clip(cond_tokens)
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)
            context = torch.cat([cond_context, uncond_context])
            context = context.repeat_interleave(num_per_image, dim=0)
        else:
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens)
            context = context.repeat_interleave(num_per_image, dim=0)
        to_idle(clip)


        
        # 2. make latents(forward VAE encoder)
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor)

            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            input_image_tensor = input_image_tensor.unsqueeze(0)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2).to(device)

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

            to_idle(encoder)
        else:
            latents = torch.cat([torch.randn(latents_shape, 
                                            generator=generator, 
                                            device=device) 
                                            for generator in generators]
                                )
        

        # 3. controlnet embedding forward
        # check model is list or one
        controlnet_embedding_models = models["controlnet_embedding"]
        control_embedding_latent_list = []

        if not isinstance(control_image, list): 
            control_image = [control_image] * len(controlnet_embedding_models)
        if not isinstance(controlnet_embedding_models, list):
            controlnet_embedding_models = [controlnet_embedding_models]
        
        # forward
        for idx, controlnet_embedding_model in enumerate(controlnet_embedding_models):
            control_image_tensor = control_image[idx].resize((WIDTH, HEIGHT))
            control_image_tensor = np.array(control_image_tensor)
            control_image_tensor = torch.tensor(control_image_tensor)
            control_image_tensor = rescale(control_image_tensor, (0, 255), (0, 1))
            control_image_tensor = control_image_tensor.unsqueeze(0)
            control_image_tensor = control_image_tensor.permute(0, 3, 1, 2).to(device)
            control_image_tensor = control_image_tensor.repeat_interleave(num_per_image, dim=0)

            controlnet_embedding_model.to(device)
            control_embedding_latent = controlnet_embedding_model(control_image_tensor.to(dtype=dtype_map["controlnet_embedding"]))
            control_embedding_latent_list.append(control_embedding_latent)
            to_idle(controlnet_embedding_model)


        # 4. DENOISING LOOP forward
        diffusion = models["diffusion"]
        diffusion.to(device)

        # check model is list or one
        controlnet_models = models["controlnet"]
        if not isinstance(controlnet_models, list):
            controlnet_models = [controlnet_models]
        if not isinstance(controlnet_scale, list):
            controlnet_scale = [controlnet_scale] * len(controlnet_models)
        for controlnet_model in controlnet_models:
            controlnet_model.to(device)

        # LOOP start
        timesteps = tqdm(sampler.timesteps, leave=leave_tqdm)
        for i, timestep in enumerate(timesteps):

            # prepare time step embedding
            time_embedding = get_time_embedding(timestep).to(device)
            
            model_input = latents
            control_img_input = latents

            # 4-1. controlnet forward
            control_embedding_input = control_embedding_latent_list

            if do_cfg:
                control_img_input = control_img_input.repeat(2, 1, 1, 1)
                control_embedding_input = [cur.repeat(2, 1, 1, 1) for cur in control_embedding_input]
            
            controlnet_downs, controlnet_mids = None, None
            for idx, controlnet_model in enumerate(controlnet_models):
                logit_downs, logit_mids = controlnet_model(original_sample=control_img_input.to(dtype=dtype_map["controlnet"]), 
                                                                    latent=control_embedding_input[idx].to(dtype=dtype_map["controlnet"]), 
                                                                    context=context.to(dtype=dtype_map["controlnet"]), 
                                                                    time=time_embedding.to(dtype=dtype_map["controlnet"]),
                                                                    controlnet_scale=controlnet_scale[idx])
                if idx == 0:
                    controlnet_downs, controlnet_mids = logit_downs, logit_mids
                else:
                    controlnet_downs = [cur + logit_downs[i] for i, cur in enumerate(controlnet_downs)] 
                    controlnet_mids = [cur + logit_mids[i] for i, cur in enumerate(controlnet_mids)] 

            # 4-2. UNET forward
            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)
            
            model_output = diffusion(model_input.to(dtype=dtype_map["diffusion"]),
                                    context.to(dtype=dtype_map["diffusion"]),
                                    time_embedding.to(dtype=dtype_map["diffusion"]),
                                    additional_res_condition=[
                                        [cur.to(dtype=dtype_map["diffusion"]) for cur in controlnet_downs], 
                                        [cur.to(dtype=dtype_map["diffusion"]) for cur in controlnet_mids]
                                    ],
                                    **kwargs
            )

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            
            # 4-3. scheduler step
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)
        for controlnet_model in controlnet_models:
            to_idle(controlnet_model)
        
        # LOOP end
        

        # 5. VAE decoder forward
        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents.to(dtype=dtype_map["decoder"]))

        to_idle(decoder)


        # 6. prepare outputs
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()

        return [Image.fromarray(image).resize([ORIGIN_WIDTH, ORIGIN_HEIGHTS]) for image in images]
