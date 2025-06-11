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

# Training utilities
def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"

def print_progress_bar(current, total, width=50):
    progress = current / total
    filled_width = int(width * progress)
    bar = '█' * filled_width + '░' * (width - filled_width)
    return f"[{bar}] {current}/{total} ({progress*100:.1f}%)"

# Calculate total batches for progress tracking
total_batches_per_epoch = sum(1 for _ in train_loader)
print(f"Starting training: {num_epochs} epochs, {total_batches_per_epoch} batches per epoch")
print(f"Total batches: {num_epochs * total_batches_per_epoch}")
print("-" * 80)

# Training Loop
training_start_time = time.time()
for epoch in range(num_epochs):
    # Regenerate loader
    train_loader = train_dataset.as_mlx() 

    epoch_start_time = time.time()
    total_loss = 0
    num_batches = 0
    correct = 0
    total = 0

    batch_times = []

    for batch_idx, (images, labels) in enumerate(train_loader):
        batch_start_time = time.time()

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

        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        
        # Print progress every 10 batches or on last batch
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches_per_epoch:
            current_loss = total_loss / num_batches
            current_acc = correct / total * 100
            avg_batch_time = sum(batch_times[-10:]) / len(batch_times[-10:])  # Last 10 batches
            
            progress_bar = print_progress_bar(batch_idx + 1, total_batches_per_epoch)
            batches_per_sec = 1.0 / avg_batch_time if avg_batch_time > 0 else 0
            
            print(f"Epoch {epoch+1}/{num_epochs} {progress_bar} | "
                  f"Loss: {current_loss:.4f} | Acc: {current_acc:.1f}% | "
                  f"{batches_per_sec:.1f} batch/s | {format_time(batch_time)} per batch")
                
    # Calculate epoch metrics
    epoch_time = time.time() - epoch_start_time
    avg_loss = total_loss / num_batches
    acc = correct / total
    avg_batch_time_epoch = epoch_time / total_batches_per_epoch
    
    # Save model weights
    os.makedirs("checkpoints", exist_ok=True)
    model.save_weights(f"checkpoints/vit_epoch_{epoch+1}.safetensors")
    
    # Calculate ETA
    elapsed_total = time.time() - training_start_time
    epochs_completed = epoch + 1
    avg_epoch_time = elapsed_total / epochs_completed
    remaining_epochs = num_epochs - epochs_completed
    eta = remaining_epochs * avg_epoch_time
    
    print("=" * 80)
    print(f"EPOCH {epoch+1}/{num_epochs} COMPLETE:")
    print(f"  Loss: {avg_loss:.4f} | Accuracy: {acc*100:.2f}%")
    print(f"  Epoch Time: {format_time(epoch_time)} | Avg Batch Time: {avg_batch_time_epoch:.3f}s")
    print(f"  Total Elapsed: {format_time(elapsed_total)} | ETA: {format_time(eta)}")
    print(f"  Samples/sec: {total/epoch_time:.1f}")
    print("=" * 80)
