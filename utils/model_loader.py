import torch
from networks.clip.clip import CLIP
from networks.clip.clip_image_encoder import CLIPImageEncoder
from networks.vae.encoder import VAE_Encoder
from networks.vae.decoder import VAE_Decoder
from networks.diffusion import Diffusion

def load_diffusion_model(state_dict=None, dtype=torch.float16, **kwargs):
    encoder = VAE_Encoder().to(dtype)
    decoder = VAE_Decoder().to(dtype)

    if kwargs.get("is_inpaint") == True:
        diffusion = Diffusion(in_channels=9, **kwargs).to(dtype)
    else:
        diffusion = Diffusion(in_channels=4, **kwargs).to(dtype)

    if state_dict is not None:
        encoder.load_state_dict(state_dict['encoder'], strict=True)
        decoder.load_state_dict(state_dict['decoder'], strict=True)

        if kwargs.get("clip_train") == True:
            clip = CLIP(n_vocab=49408).to(dtype=kwargs.get("clip_dtype"))
            clip.load_state_dict(state_dict['clip'], strict=True)
        elif kwargs.get("clip_image_encoder") == True:
            clip = CLIPImageEncoder(from_pretrained=kwargs.get("clip_image_encoder_from_pretrained")).to(dtype=kwargs.get("clip_dtype"))
        else:
            clip = CLIP(n_vocab=49408).to(dtype)
            clip.load_state_dict(state_dict['clip'], strict=True)

        if kwargs.get("is_lora") == True:
            diffusion.load_state_dict(state_dict['diffusion'], strict=False)
            if state_dict.get("lora") is not None:
                diffusion.load_state_dict(state_dict['lora'], strict=False)
        else:
            diffusion.load_state_dict(state_dict['diffusion'], strict=True)
        

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }


from networks.controlnet.controlnet import Controlnet, ControlNetConditioningEmbedding

def load_controlnet_model(state_dict=None, dtype=torch.float16, **kwargs):
    if not isinstance(state_dict, list):
        state_dict = [state_dict]
    state_dict_list = state_dict

    controlnet_list, controlnet_embedding_list = [], [] 
    for state_dict in state_dict_list:
        
        controlnet = Controlnet(4, **kwargs).to(dtype)
        controlnet_embedding = ControlNetConditioningEmbedding(320, 3).to(dtype)
    
        if state_dict is not None:
            controlnet.load_state_dict(state_dict["controlnet"])
            controlnet_embedding.load_state_dict(state_dict["embedding"])

        controlnet_list.append(controlnet)
        controlnet_embedding_list.append(controlnet_embedding)
    
    return {
        'controlnet': controlnet_list,
        'controlnet_embedding': controlnet_embedding_list,
    }
