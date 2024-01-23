from torch import nn
from torch.nn import functional as F
from diffusion import *



def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ControlNetConditioningEmbedding(nn.Module):

    def __init__(
        self,
        conditioning_embedding_channels: int = 320,
        conditioning_channels: int = 3,
        block_out_channels = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding



class Controlnet(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.time_embed = TimeEmbedding(320)

        self.encoders = nn.ModuleList([
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(in_channels, 320, kernel_size=3, padding=1)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 320, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 640, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(1280, 1280), 
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_AttentionBlock(8, 160), 
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(1280, 1280), 
        )

        self.bottleneck_out = zero_module(nn.Conv2d(1280, 1280, kernel_size=1, padding="valid"))

        self.zero_convs = nn.ModuleList([
            zero_module(nn.Conv2d(320, 320, kernel_size=1, padding="valid")),
            zero_module(nn.Conv2d(320, 320, kernel_size=1, padding="valid")),
            zero_module(nn.Conv2d(320, 320, kernel_size=1, padding="valid")),
            zero_module(nn.Conv2d(320, 320, kernel_size=1, padding="valid")),
            zero_module(nn.Conv2d(640, 640, kernel_size=1, padding="valid")),
            zero_module(nn.Conv2d(640, 640, kernel_size=1, padding="valid")),
            zero_module(nn.Conv2d(640, 640, kernel_size=1, padding="valid")),
            zero_module(nn.Conv2d(1280, 1280, kernel_size=1, padding="valid")),
            zero_module(nn.Conv2d(1280, 1280, kernel_size=1, padding="valid")),
            zero_module(nn.Conv2d(1280, 1280, kernel_size=1, padding="valid")),
            zero_module(nn.Conv2d(1280, 1280, kernel_size=1, padding="valid")),
            zero_module(nn.Conv2d(1280, 1280, kernel_size=1, padding="valid")),
        ])

    def forward(self, original_sample, latent, context, time):
        time = self.time_embed(time)
        
        sample = self.encoders[0](original_sample, context, time)
        latent = sample + latent

        skip_connections = [latent]
        for layers in self.encoders[1:]:
            latent = layers(latent, context, time)
            skip_connections.append(latent)

        controlnet_outs = []
        for skip_connection, zero_conv in zip(skip_connections, self.zero_convs):
            controlnet_outs.append(zero_conv(skip_connection))

        latent = self.bottleneck(latent, context, time)
        latent = self.bottleneck_out(latent)

        controlnet_downs = controlnet_outs[::-1]
        controlnet_mids = [latent]
        
        return controlnet_downs, controlnet_mids
