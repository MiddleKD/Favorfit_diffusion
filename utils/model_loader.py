import torch
from ..models.clip.clip import CLIP
from ..models.vae.encoder import VAE_Encoder
from ..models.vae.decoder import VAE_Decoder
from ..models.diffusion import Diffusion

def load_diffusion_model(state_dict=None, dtype=torch.float16, **kwargs):

    encoder = VAE_Encoder().to(dtype)
    decoder = VAE_Decoder().to(dtype)
    clip = CLIP().to(dtype)

    if kwargs.get("is_inpaint") == True:
        diffusion = Diffusion(in_channels=9, **kwargs).to(dtype)
    else:
        diffusion = Diffusion(in_channels=4, **kwargs).to(dtype)

    if state_dict is not None:
        encoder.load_state_dict(state_dict['encoder'], strict=True)
        decoder.load_state_dict(state_dict['decoder'], strict=True)
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


from ..models.controlnet.controlnet import Controlnet, ControlNetConditioningEmbedding

def load_controlnet_model(state_dict=None, dtype=torch.float16):
    
    controlnet = Controlnet(4).to(dtype)
    controlnet_embedding = ControlNetConditioningEmbedding(320, 3).to(dtype)
    
    if state_dict is not None:
        controlnet.load_state_dict(state_dict["controlnet"])
        controlnet_embedding.load_state_dict(state_dict["embedding"])

    return {
        'controlnet': controlnet,
        'controlnet_embedding': controlnet_embedding,
    }


from ..models.embedding.color_palette_embedding import ColorPaletteEmbedding, ColorPaletteTimestepEmbedding
def load_color_palette_embedding_model(state_dict=None, num_features=119, n_embd=768, n_embd_ts=320, dtype=torch.float16):
    
    colorpalette_model = ColorPaletteEmbedding(num_features, n_embd).to(dtype)
    colorpalette_timestep_model = ColorPaletteTimestepEmbedding(num_features, n_embd_ts).to(dtype)
    
    if state_dict is not None:
        colorpalette_model.load_state_dict(state_dict["color_palette_embedding"])
        colorpalette_timestep_model.load_state_dict(state_dict["color_palette_timestep_embedding"])

    return {
        'color_palette_embedding': colorpalette_model,
        'color_palette_timestep_embedding': colorpalette_timestep_model
    }
