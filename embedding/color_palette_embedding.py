from torch import nn
import torch


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ResBlock(nn.Module):
    def __init__(self, in_features, out_features, hidden_features):
        super().__init__()

        self.layer1 = nn.Linear(in_features, hidden_features)
        self.layer2 = nn.Linear(hidden_features, hidden_features)
        self.layer3 = nn.Linear(hidden_features, out_features)
        
        if in_features == out_features:
            self.skip_layer = nn.Identity()
        else:
            self.skip_layer = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        residue = x
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        residue = self.skip_layer(residue)

        return x + residue  


class ColorPaletteEmbedding(nn.Module):
    def __init__(self, in_features=119, n_embd=768):
        super().__init__()

        self.n_embd = n_embd

        self.cl_encoder = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.SELU(),
            nn.LayerNorm(512),
            ResBlock(512, 1024, 1024),
            nn.SELU(),
            nn.LayerNorm(1024),
            ResBlock(1024, 1024, 2048)
        )
        
        self.out_layer = nn.Linear(1024, n_embd)

    def forward(self, x):
        x = self.cl_encoder(x)
        x = self.out_layer(x)

        if len(x.shape) == 2:
            x = x.reshape([-1,1,self.n_embd])
        return x


class ColorPaletteTimestepEmbedding(nn.Module):
    def __init__(self, in_features=119, n_embd=320):
        super().__init__()

        self.cl_encoder = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.SELU(),
            nn.LayerNorm(512),
            ResBlock(512, 1024, 1024),
            nn.SELU(),
            nn.LayerNorm(1024),
            ResBlock(1024, 512, 1024)
        )
        
        self.out_layer = zero_module(nn.Linear(512, n_embd))

    def forward(self, x):
        x = self.cl_encoder(x)
        x = self.out_layer(x)
        return x


if __name__ == "__main__":
    temp_input = torch.randn([3, 119])
    model = ColorPaletteEmbedding()

    output = model(temp_input)
    print(output.shape)

    model = ColorPaletteTimestepEmbedding()

    output = model(temp_input)
    print(output.shape)