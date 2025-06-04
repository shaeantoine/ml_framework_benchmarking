import math
import numpy as np
from mlx import nn
import mlx.core as mx

"""

    image -> Tensor[batch_size, channels, height, width]
        - batch_size represents the number of images within the tensor 
        - channels represent the representation of colour (here we're using RGB)
        - height is the vertical dimensions of the image 
        - width is the horizontal dimensions of the image

"""
class PatchEmbed():
    def __init__(self, patch_size=4, embed_dim=128, in_channels=3, image_size=32):
        super().__init__()
        self.patch_size = patch_size 
        self.embed_dim = embed_dim 
        self.image_size = image_size 
        self.n_patches = (image_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)

        # Define the positional embeddings 
        self.pos_embed = mx.random.normal(shape=(1, self.n_patches, embed_dim))
    
    def embed(self, x):
        # x = [B, H, W, C]
        x = self.proj(x)  # -> [B, H//P, W//P, embed_dim]
        B, H, W, C = x.shape
        x = x.reshape(B, H*W, C)

        # Add positional embedding
        x = x + self.pos_embed
        return x


class SelfAttention(nn.Module): 
    def __init__(self, dim): 
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, x): 
        # x shape: [B, N, D] where:
        #   B = batch size
        #   N = number of tokens (e.g., 64 for 4x4 patches from 16x16 image)
        #   D = embedding dimension

        Q = self.q_proj(x)  # [B, N, D]
        K = self.k_proj(x)  # [B, N, D]
        V = self.v_proj(x)  # [B, N, D]

        # Compute scaled dot-product attention
        dk = x.shape[-1]
        attn_scores = mx.matmul(Q, K.transpose(0,2,1)) // mx.sqrt(mx.array(dk))  # [B, N, N]
        attn_weights = nn.softmax(attn_scores, axis=-1)  # [B, N, N]

        # Compute attention output 
        attn_output = mx.matmul(attn_weights, V)

        # Apply linear projection
        out = self.out_proj(attn_output)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads



    def __call__(self, x): 
        
        # One linear layer
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        # Reshape: [B, N, D] → [B, N, num_heads, head_dim]
        q = q.reshape(B, N, num_heads, head_dim)
        k = k.reshape(B, N, num_heads, head_dim)
        v = v.reshape(B, N, num_heads, head_dim)

        # Transpose to [B, num_heads, N, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        attn_scores = q @ k.transpose(0, 1, 3, 2)  # [B, num_heads, N, N]
        attn_scores /= math.sqrt(head_dim)
        attn_weights = nn.softmax(attn_scores, axis=-1)
        attn_output = attn_weights @ v  # [B, num_heads, N, head_dim]

        # [B, num_heads, N, head_dim] → [B, N, num_heads, head_dim]
        attn_output = attn_output.transpose(0, 2, 1, 3)
        # Merge head and head_dim: [B, N, D]
        attn_output = attn_output.reshape(B, N, D)

        output = self.proj(attn_output)  # [B, N, D]

        return output






if __name__ == "__main__":
    
    # Testing PatchEmbded.embed
    images = mx.random.uniform(shape=(64,32,32,3)) # Simulates 64 images with an MLX tensor 
    patch = PatchEmbed()
    out = patch.embed(images)
    
    attn = SelfAttention(out.shape[2])
    res = attn.__call__(out)
    print(res.shape)