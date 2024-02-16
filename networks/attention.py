import torch
from torch import nn
from torch.nn import functional as F
import math
from networks.lora.lora import get_lora_layers


class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True, **kwargs):
        super().__init__()

        # This combines the Wq, Wk and Wv matrices into one matrix
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        # This one represents the Wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

        self.is_lora = kwargs.get('is_lora')

        if self.is_lora == True:
            self.k_lora_down, self.k_lora_up = get_lora_layers(d_embed, d_embed, rank=4)
            self.q_lora_down, self.q_lora_up = get_lora_layers(d_embed, d_embed, rank=4)
            self.v_lora_down, self.v_lora_up = get_lora_layers(d_embed, d_embed, rank=4)
            self.out_lora_down, self.out_lora_up = get_lora_layers(d_embed, d_embed, rank=4)

    def forward(self, x, causal_mask=False, **kwargs):
        # x: # (Batch_Size, Seq_Len, Dim)

        # (Batch_Size, Seq_Len, Dim)
        input_shape = x.shape 
        
        # (Batch_Size, Seq_Len, Dim)
        batch_size, sequence_length, d_embed = input_shape 

        # (Batch_Size, Seq_Len, H, Dim / H)
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head) 

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensor of shape (Batch_Size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch_Size, H, Seq_Len, Dim) @ (Batch_Size, H, Dim, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)
        
        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1) 
            # Fill the upper triangle with -inf
            weight.masked_fill_(mask, -float("inf")) 
        
        # Divide by d_k (Dim / H). 
        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight /= math.sqrt(self.d_head) 

        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = F.softmax(weight, dim=-1) 

        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        output = weight @ v

        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
        output = output.transpose(1, 2) 
        
        # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)
        output = output.reshape(input_shape) 

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.out_proj(output) 
        
        # (Batch_Size, Seq_Len, Dim)

        if self.is_lora == True:
            if kwargs.get('lora_scale') == None:
                self.lora_scale = 1.0
            else:
                self.lora_scale = kwargs.get('lora_scale')

            if self.q_lora_down.weight.dtype == torch.float16:
                lora_weight_type = torch.float16
            else:
                lora_weight_type = torch.float32
            
            q_lora = self.q_lora_up(self.q_lora_down(x.to(dtype=lora_weight_type))).view(interim_shape).transpose(1, 2)
            q_lora = q.to(dtype=lora_weight_type) + self.lora_scale * q_lora
            k_lora = self.k_lora_up(self.k_lora_down(x.to(dtype=lora_weight_type))).view(interim_shape).transpose(1, 2)
            k_lora = k.to(dtype=lora_weight_type) + self.lora_scale * k_lora
            v_lora = self.v_lora_up(self.v_lora_down(x.to(dtype=lora_weight_type))).view(interim_shape).transpose(1, 2)
            v_lora = v.to(dtype=lora_weight_type) + self.lora_scale * v_lora

            weight = q_lora @ k_lora.transpose(-1, -2)
            weight /= math.sqrt(self.d_head) 
            weight = F.softmax(weight, dim=-1)
            output_lora = weight @ v_lora
            output_lora = output_lora.transpose(1, 2) 
            output_lora = output_lora.reshape(input_shape) 

            output_lora = self.out_lora_up(self.out_lora_down(output_lora))
            
            return output + self.lora_scale * output_lora.to(dtype=output.dtype)

        return output


class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True, **kwargs):
        super().__init__()

        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

        self.is_lora = kwargs.get('is_lora')

        if self.is_lora == True:
            self.k_lora_down, self.k_lora_up = get_lora_layers(d_cross, d_embed, rank=4)
            self.q_lora_down, self.q_lora_up = get_lora_layers(d_embed, d_embed, rank=4)
            self.v_lora_down, self.v_lora_up = get_lora_layers(d_cross, d_embed, rank=4)
            self.out_lora_down, self.out_lora_up = get_lora_layers(d_embed, d_embed, rank=4)
    
    def forward(self, x, y, **kwargs):
        # x (latent): # (Batch_Size, Seq_Len_Q, Dim_Q)
        # y (context): # (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        q = self.q_proj(x)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        k = self.k_proj(y)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        v = self.v_proj(y)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        q = q.view(interim_shape).transpose(1, 2) 
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        k = k.view(interim_shape).transpose(1, 2) 
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        v = v.view(interim_shape).transpose(1, 2) 
        
        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) -> (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = q @ k.transpose(-1, -2)
        
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight /= math.sqrt(self.d_head)
        
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = F.softmax(weight, dim=-1)
        
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV) @ (Batch_Size, H, Seq_Len_KV, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        output = weight @ v
        
        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H)
        output = output.transpose(1, 2).contiguous()
        
        # (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = output.view(input_shape)
        
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = self.out_proj(output)

        # (Batch_Size, Seq_Len_Q, Dim_Q)

        if self.is_lora == True:
            if kwargs.get('lora_scale') == None:
                self.lora_scale = 1.0
            else:
                self.lora_scale = kwargs.get('lora_scale')

            if self.q_lora_down.weight.dtype == torch.float16:
                lora_weight_type = torch.float16
            else:
                lora_weight_type = torch.float32

            q_lora = self.q_lora_up(self.q_lora_down(x.to(dtype=lora_weight_type))).view(interim_shape).transpose(1, 2)
            q_lora = q.to(dtype=lora_weight_type) + self.lora_scale * q_lora
            k_lora = self.k_lora_up(self.k_lora_down(y.to(dtype=lora_weight_type))).view(interim_shape).transpose(1, 2)
            k_lora = k.to(dtype=lora_weight_type) + self.lora_scale * k_lora
            v_lora = self.v_lora_up(self.v_lora_down(y.to(dtype=lora_weight_type))).view(interim_shape).transpose(1, 2)
            v_lora = v.to(dtype=lora_weight_type) + self.lora_scale * v_lora
    
            weight = q_lora @ k_lora.transpose(-1, -2)
            weight /= math.sqrt(self.d_head) 
            weight = F.softmax(weight, dim=-1)
            output_lora = weight @ v_lora
            output_lora = output_lora.transpose(1, 2).contiguous()
            output_lora = output_lora.view(input_shape)

            output_lora = self.out_lora_up(self.out_lora_down(output_lora))
            
            return output + self.lora_scale * output_lora.to(dtype=output.dtype)
        
        return output