import os
import time
import mlx.nn as nn
import mlx.core as mx
from models.vit_mlx import ViT_MLX
from mlx.optimizers import Adam
#from nn.Module import save_weights
from data.data_load import DiskCachedDataset

# Training Config
num_classes = 20
image_size = 224
patch_size = 16
batch_size = 128
num_epochs = 10
lr = 3e-4

# Load Dataset 
train_dataset = DiskCachedDataset(split="train", size=(image_size, image_size))
train_loader = train_dataset.as_mlx()

# Instantiate Model 
model = ViT_MLX(
    img_size=image_size,
    patch_size=patch_size,
    num_classes=num_classes,
    embed_dim=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4.0,
    dropout=0.1
)

# Optimizer 
params = model.parameters()
optimizer = Adam(learning_rate=lr)

# Loss Function
def compute_loss_and_grads(model, images, labels):
    def loss_fn(params):
        model.update(params)
        logits = model(images)
        loss = nn.losses.cross_entropy(logits, labels)
        return loss, logits
    (loss, logits), grads = mx.value_and_grad(loss_fn, has_aux=True)(model.parameters())
    return loss, logits, grads

# Training Loop
for epoch in range(num_epochs):
    total_loss = 0
    num_batches = 0
    correct = 0
    total = 0

    start_time = time.time()

    for images, labels in train_loader:
        loss, logits, grads = compute_loss_and_grads(model, images, labels)
        params = optimizer.update(params, grads) 

        total_loss += loss.item()
        num_batches += 1

        preds = mx.argmax(logits, axis=-1)
        correct += mx.sum(preds == labels).item()
        total += labels.size

    avg_loss = total_loss / num_batches
    acc = correct / total
    
    nn.save_weights(f"checkpoints/vit_epoch_{epoch+1}.safetensors", params)

    print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Accuracy: {acc*100:.2f}% | Time: {time.time() - start_time:.2f}s")
