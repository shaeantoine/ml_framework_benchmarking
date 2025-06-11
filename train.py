import os
import time
import mlx.nn as nn
import mlx.core as mx
from mlx.optimizers import Adam
from models.vit_mlx import ViT_MLX
from data.data_load import DiskCachedDataset

# Training Config
num_classes = 20
image_size = 224
patch_size = 16
batch_size = 128
num_epochs = 10
lr = 3e-4

# Load Dataset 
train_dataset = DiskCachedDataset(split="train", cache_dir="data/processed")
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
optimizer = Adam(learning_rate=lr)

# Loss Function
def compute_loss(model, images, labels):
    logits = model(images)
    return nn.losses.cross_entropy(logits, labels, reduction='mean')

loss_and_grad_fn = nn.value_and_grad(model, compute_loss)

# Training Loop
for epoch in range(num_epochs):
    total_loss = 0
    num_batches = 0
    correct = 0
    total = 0

    start_time = time.time()

    for images, labels in train_loader:
        images = mx.transpose(images, (0, 2, 3, 1)) # Convert format

        # Forward pass and compute gradients
        loss, grads = loss_and_grad_fn(model, images, labels)
        
        # Update model parameters
        optimizer.update(model, grads)
        
        # Compute predictions for accuracy
        logits = model(images)
        preds = mx.argmax(logits, axis=-1)

        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        correct += mx.sum(preds == labels).item()
        total += labels.size

    avg_loss = total_loss / num_batches
    acc = correct / total
    
    model.save_weights(f"checkpoints/vit_epoch_{epoch+1}.safetensors")

    print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Accuracy: {acc*100:.2f}% | Time: {time.time() - start_time:.2f}s")
