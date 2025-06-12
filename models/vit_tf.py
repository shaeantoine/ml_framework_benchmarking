import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class PatchEmbed(layers.Layer):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=384, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding='valid')

    def call(self, x):
        x = self.proj(x)  # (B, H/P, W/P, D)
        
        B = tf.shape(x)[0]
        H_patches, W_patches, D = x.shape[1], x.shape[2], x.shape[3]
        x = tf.reshape(x, [B, H_patches * W_patches, D])  # (B, N, D) where N = H_patches * W_patches
        
        return x


class MLP(layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0., **kwargs):
        super().__init__(**kwargs)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = layers.Dense(hidden_features)
        self.act = layers.Activation('gelu')
        self.fc2 = layers.Dense(out_features)
        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        return x


class Attention(layers.Layer):
    def __init__(self, dim, num_heads=6, dropout=0., **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = layers.Dense(dim * 3)
        self.proj = layers.Dense(dim)
        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=None):
        B, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [B, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn, training=training)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [B, N, C])
        x = self.proj(x)
        x = self.dropout(x, training=training)
        return x


class TransformerBlock(layers.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0., **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization()
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = layers.LayerNormalization()
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)

    def call(self, x, training=None):
        x = x + self.attn(self.norm1(x), training=training)
        x = x + self.mlp(self.norm2(x), training=training)
        return x


class ViT_TF(keras.Model):
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
        **kwargs
    ):
        super().__init__(**kwargs)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        token_count = num_patches + 1 + int(use_distill_token)

        # Class and distillation token
        self.cls_token = self.add_weight(
            name='cls_token',
            shape=(1, 1, embed_dim),
            initializer='random_normal',
            trainable=True
        )
        self.use_distill_token = use_distill_token
        if use_distill_token:
            self.dist_token = self.add_weight(
                name='dist_token',
                shape=(1, 1, embed_dim),
                initializer='random_normal',
                trainable=True
            )
        else:
            self.dist_token = None

        # Positional embedding
        self.pos_embed = self.add_weight(
            name='pos_embed',
            shape=(1, token_count, embed_dim),
            initializer='random_normal',
            trainable=True
        )
        self.dropout = layers.Dropout(dropout)

        self.blocks = [
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) 
            for _ in range(depth)
        ]
        self.norm = layers.LayerNormalization()

        self.head = layers.Dense(num_classes)
        if use_distill_token:
            self.head_dist = layers.Dense(num_classes)
        else:
            self.head_dist = None

    def call(self, x, training=None):
        B = tf.shape(x)[0]
        x = self.patch_embed(x)

        cls_tokens = tf.broadcast_to(self.cls_token, [B, 1, tf.shape(x)[-1]])
        if self.use_distill_token:
            dist_tokens = tf.broadcast_to(self.dist_token, [B, 1, tf.shape(x)[-1]])
            x = tf.concat([cls_tokens, dist_tokens, x], axis=1)
        else:
            x = tf.concat([cls_tokens, x], axis=1)

        x = x + self.pos_embed
        x = self.dropout(x, training=training)
        
        for block in self.blocks:
            x = block(x, training=training)
        
        x = self.norm(x)

        if self.use_distill_token:
            cls_out = self.head(x[:, 0])
            dist_out = self.head_dist(x[:, 1])
            return (cls_out + dist_out) / 2
        else:
            return self.head(x[:, 0])