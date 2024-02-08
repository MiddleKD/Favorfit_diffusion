if __name__ == "__main__":
    import sys, os
    sys.path.append(os.getcwd())

import torch
from torch import nn
from torch.nn import functional as F
from models.attention import SelfAttention
from models.model_utils import zero_module


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
    def __init__(self, in_features=3, n_embd=768):
        super().__init__()

        self.n_embd = n_embd
        self.position_embedding = nn.Parameter(torch.zeros(n_embd))

        self.pre_layer = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, n_embd),
        )

        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(12, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        x = self.pre_layer(x)
        x += self.position_embedding

        residue = x
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=False)
        x += residue

        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x)
        x = self.linear_2(x)
        x += residue
        
        return x


class ColorPaletteTimestepEmbedding(nn.Module):
    def __init__(self, in_features=12, n_embd=320):
        super().__init__()

        self.position_embedding = nn.Parameter(torch.zeros(in_features))
        self.cl_encoder = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.Linear(128, 320),
        )
        
        self.out_layer = nn.Linear(n_embd, n_embd)

    def forward(self, x, time):
        x = x.reshape([x.size(0), -1])
        x += self.position_embedding
        x = self.cl_encoder(x)
        x += time
        x = self.out_layer(x)
        return x
    

if __name__ == "__main__":

    temp_input = torch.randn([3, 4, 3])
    model = ColorPaletteEmbedding()

    output = model(temp_input)
    print(output.shape)

    model = ColorPaletteTimestepEmbedding()
    time = torch.randn([3,320])
    output = model(temp_input, time)
    print(output.shape)