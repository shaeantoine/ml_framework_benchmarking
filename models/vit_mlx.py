import math
import numpy as np
from mlx import nn
import mlx.core as mx

# """

#     image -> Tensor[batch_size, channels, height, width]
#         - batch_size represents the number of images within the tensor 
#         - channels represent the representation of colour (here we're using RGB)
#         - height is the vertical dimensions of the image 
#         - width is the horizontal dimensions of the image

# """
# class PatchEmbed():
#     def __init__(self, patch_size=4, embed_dim=128, in_channels=3, image_size=32):
#         super().__init__()
#         self.patch_size = patch_size 
#         self.embed_dim = embed_dim 
#         self.image_size = image_size 
#         self.n_patches = (image_size // patch_size) ** 2

#         self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)

#         # Define the positional embeddings 
#         self.pos_embed = mx.random.normal(shape=(1, self.n_patches, embed_dim))
    
#     def embed(self, x):
#         # x = [B, H, W, C]
#         x = self.proj(x)  # -> [B, H//P, W//P, embed_dim]
#         B, H, W, C = x.shape
#         x = x.reshape(B, H*W, C)

#         # Add positional embedding
#         x = x + self.pos_embed
#         return x


# class SelfAttention(nn.Module): 
#     def __init__(self, dim): 
#         super().__init__()
#         self.q_proj = nn.Linear(dim, dim)
#         self.k_proj = nn.Linear(dim, dim)
#         self.v_proj = nn.Linear(dim, dim)
#         self.out_proj = nn.Linear(dim, dim)

#     def __call__(self, x): 
#         # x shape: [B, N, D] where:
#         #   B = batch size
#         #   N = number of tokens (e.g., 64 for 4x4 patches from 16x16 image)
#         #   D = embedding dimension

#         Q = self.q_proj(x)  # [B, N, D]
#         K = self.k_proj(x)  # [B, N, D]
#         V = self.v_proj(x)  # [B, N, D]

#         # Compute scaled dot-product attention
#         dk = x.shape[-1]
#         attn_scores = mx.matmul(Q, K.transpose(0,2,1)) // mx.sqrt(mx.array(dk))  # [B, N, N]
#         attn_weights = nn.softmax(attn_scores, axis=-1)  # [B, N, N]

#         # Compute attention output 
#         attn_output = mx.matmul(attn_weights, V)

#         # Apply linear projection
#         out = self.out_proj(attn_output)

#         return out


# class MultiHeadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads



#     def __call__(self, x): 
        
#         # One linear layer
#         qkv = self.qkv(x)
#         q, k, v = mx.split(qkv, 3, axis=-1)

#         # Reshape: [B, N, D] → [B, N, num_heads, head_dim]
#         q = q.reshape(B, N, num_heads, head_dim)
#         k = k.reshape(B, N, num_heads, head_dim)
#         v = v.reshape(B, N, num_heads, head_dim)

#         # Transpose to [B, num_heads, N, head_dim]
#         q = q.transpose(0, 2, 1, 3)
#         k = k.transpose(0, 2, 1, 3)
#         v = v.transpose(0, 2, 1, 3)

#         attn_scores = q @ k.transpose(0, 1, 3, 2)  # [B, num_heads, N, N]
#         attn_scores /= math.sqrt(head_dim)
#         attn_weights = nn.softmax(attn_scores, axis=-1)
#         attn_output = attn_weights @ v  # [B, num_heads, N, head_dim]

#         # [B, num_heads, N, head_dim] → [B, N, num_heads, head_dim]
#         attn_output = attn_output.transpose(0, 2, 1, 3)
#         # Merge head and head_dim: [B, N, D]
#         attn_output = attn_output.reshape(B, N, D)

#         output = self.proj(attn_output)  # [B, N, D]

#         return output






# if __name__ == "__main__":
    
#     # Testing PatchEmbded.embed
#     images = mx.random.uniform(shape=(64,32,32,3)) # Simulates 64 images with an MLX tensor 
#     patch = PatchEmbed()
#     out = patch.embed(images)
    
#     attn = SelfAttention(out.shape[2])
#     res = attn.__call__(out)
#     print(res.shape)


# Utility functions
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def __call__(self, x):
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, D, N)
        x = x.transpose(0, 2, 1)  # (B, N, D)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=6, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        attn = nn.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x = mx.matmul(attn, v)
        x = x.transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.,
        use_distill_token=False,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = mx.zeros((1, 1, embed_dim))
        if use_distill_token:
            self.dist_token = mx.zeros((1, 1, embed_dim))
        else:
            self.dist_token = None

        self.pos_embed = mx.zeros((1, num_patches + 1 + int(use_distill_token), embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes)
        if use_distill_token:
            self.head_dist = nn.Linear(embed_dim, num_classes)
        else:
            self.head_dist = None

    def __call__(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = mx.broadcast_to(self.cls_token, (B, 1, x.shape[-1]))
        if self.dist_token is not None:
            dist_tokens = mx.broadcast_to(self.dist_token, (B, 1, x.shape[-1]))
            x = mx.concatenate([cls_tokens, dist_tokens, x], axis=1)
        else:
            x = mx.concatenate([cls_tokens, x], axis=1)

        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.norm(x)

        if self.dist_token is not None:
            cls_out = self.head(x[:, 0])
            dist_out = self.head_dist(x[:, 1])
            return (cls_out + dist_out) / 2
        else:
            return self.head(x[:, 0])
