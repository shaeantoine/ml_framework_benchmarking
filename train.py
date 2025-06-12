import os
import time
import mlx.nn as nn
import mlx.core as mx
from mlx.optimizers import Adam
from telemetry import ViTTelemetry
from models.vit_mlx import ViT_MLX
from data.data_load import DiskCachedDataset

# Training Config
num_classes = 20
image_size = 224
patch_size = 16
batch_size = 128
num_epochs = 10
lr = 3e-4

# Initialize telemetry 
telemetry = ViTTelemetry(framework="mlx")
print("Telemetry system initialized for MLX framework")
telemetry.print_summary()

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

def print_telemetry_summary(telemetry, epoch, batch_idx, current_loss, current_acc):
    # Print real-time telemetry data
    if telemetry.metrics_history:
        latest_metrics = telemetry.metrics_history[-1]
        print(f"Memory: {latest_metrics.memory_used_gb:.2f}GB | "
              f"Peak: {latest_metrics.peak_memory_gb:.2f}GB | "
              f"CPU: {latest_metrics.cpu_utilization_percent:.1f}% | "
              f"Power: {latest_metrics.power_draw_watts:.1f}W | "
              f"Temp: {latest_metrics.temperature_celsius:.1f}°C")

# Calculate total batches for progress tracking
total_batches_per_epoch = sum(1 for _ in train_loader)
print(f"Starting training: {num_epochs} epochs, {total_batches_per_epoch} batches per epoch")
print(f"Total batches: {num_epochs * total_batches_per_epoch}")
print("-" * 80)

# Start background monitoring for continuous system metrics
telemetry.start_background_monitoring(interval=0.5)  # Monitor every 500ms

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
    epoch_forward_times = []
    epoch_backward_times = []

    for batch_idx, (images, labels) in enumerate(train_loader):
        batch_start_time = time.time()

        # Get actual batch size (might be different for last batch)
        actual_batch_size = images.shape[0]
        sequence_length = (image_size // patch_size) ** 2

        images = mx.transpose(images, (0, 2, 3, 1)) # Convert format

       # Forward pass with telemetry
        with telemetry.measure_operation("forward", actual_batch_size, sequence_length):
            loss, grads = loss_and_grad_fn(model, images, labels)
        
        # Backward pass (parameter update) with telemetry
        with telemetry.measure_operation("backward", actual_batch_size, sequence_length):
            optimizer.update(model, grads)
        
        # Inference for accuracy (separate measurement)
        with telemetry.measure_operation("inference", actual_batch_size, sequence_length):
            logits = model(images)
            preds = mx.argmax(logits, axis=-1)

        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        correct += mx.sum(preds == labels).item()
        total += labels.size

        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        
        # Collect timing stats for epoch summary
        if telemetry.metrics_history:
            latest_forward = next((m.forward_time_ms for m in reversed(telemetry.metrics_history) 
                                 if m.forward_time_ms > 0), 0)
            latest_backward = next((m.backward_time_ms for m in reversed(telemetry.metrics_history) 
                                  if m.backward_time_ms > 0), 0)
            if latest_forward > 0:
                epoch_forward_times.append(latest_forward)
            if latest_backward > 0:
                epoch_backward_times.append(latest_backward)
        
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
            
            # Print telemetry data
            print_telemetry_summary(telemetry, epoch, batch_idx, current_loss, current_acc)
                
    # Calculate epoch metrics
    epoch_time = time.time() - epoch_start_time
    avg_loss = total_loss / num_batches
    acc = correct / total
    avg_batch_time_epoch = epoch_time / total_batches_per_epoch

    # Calculate epoch telemetry stats
    avg_forward_time = sum(epoch_forward_times) / len(epoch_forward_times) if epoch_forward_times else 0
    avg_backward_time = sum(epoch_backward_times) / len(epoch_backward_times) if epoch_backward_times else 0
    
    # Save model weights
    os.makedirs("checkpoints", exist_ok=True)
    model.save_weights(f"checkpoints/vit_epoch_{epoch+1}.safetensors")

    # Save telemetry data for this epoch
    os.makedirs("telemetry", exist_ok=True)
    telemetry.save_results(f"telemetry/epoch_{epoch+1}_telemetry.json")
    
    # Calculate ETA
    elapsed_total = time.time() - training_start_time
    epochs_completed = epoch + 1
    avg_epoch_time = elapsed_total / epochs_completed
    remaining_epochs = num_epochs - epochs_completed
    eta = remaining_epochs * avg_epoch_time

    # Get current system metrics
    current_memory = telemetry._get_memory_usage()
    current_power = telemetry._get_power_draw()
    current_temp = telemetry._get_temperature()
    
    print("=" * 80)
    print(f"EPOCH {epoch+1}/{num_epochs} COMPLETE:")
    print(f"  Training Metrics:")
    print(f"    Loss: {avg_loss:.4f} | Accuracy: {acc*100:.2f}%")
    print(f"    Samples/sec: {total/epoch_time:.1f}")
    print(f"  ")
    print(f"  Performance Metrics:")
    print(f"    Epoch Time: {format_time(epoch_time)} | Avg Batch Time: {avg_batch_time_epoch:.3f}s")
    print(f"    Avg Forward Time: {avg_forward_time:.2f}ms | Avg Backward Time: {avg_backward_time:.2f}ms")
    print(f"  ")
    print(f"  System Metrics:")
    print(f"    Memory Used: {current_memory[0]:.2f}GB / {current_memory[1]:.2f}GB")
    print(f"    Peak Memory: {current_memory[2]:.2f}GB")
    print(f"    Power Draw: {current_power:.1f}W | Temperature: {current_temp:.1f}°C")
    print(f"  ")
    print(f"  Progress:")
    print(f"    Total Elapsed: {format_time(elapsed_total)} | ETA: {format_time(eta)}")
    print("=" * 80)

telemetry.stop_background_monitoring()

# Training complete - generate comprehensive report
print("\n" + "="*80)
print("TRAINING COMPLETE - GENERATING COMPREHENSIVE TELEMETRY REPORT")
print("="*80)

# Print final telemetry summary
telemetry.print_summary()

# Save comprehensive results
telemetry.save_results("telemetry/final_training_telemetry.json")
telemetry.save_results("telemetry/final_training_telemetry.csv", format='csv')

# Generate training summary report
training_total_time = time.time() - training_start_time
summary_stats = telemetry.get_summary_stats()

print(f"\nTRAINING SUMMARY:")
print(f"Total Training Time: {format_time(training_total_time)}")
print(f"Total Measurements: {len(telemetry.metrics_history)}")
print(f"Average Throughput: {summary_stats.get('throughput_samples_per_sec', {}).get('mean', 0):.1f} samples/sec")

if summary_stats.get('forward_time_ms', {}).get('count', 0) > 0:
    forward_stats = summary_stats['forward_time_ms']
    print(f"Forward Pass Stats: {forward_stats['mean']:.2f}ms ± {forward_stats['std']:.2f}ms")

if summary_stats.get('backward_time_ms', {}).get('count', 0) > 0:
    backward_stats = summary_stats['backward_time_ms']
    print(f"Backward Pass Stats: {backward_stats['mean']:.2f}ms ± {backward_stats['std']:.2f}ms")

print(f"\nTelemetry files saved:")
print(f"  - telemetry/final_training_telemetry.json (comprehensive data)")
print(f"  - telemetry/final_training_telemetry.csv (metrics data)")
print(f"  - telemetry/epoch_N_telemetry.json (per-epoch data)")

print(f"\nCheckpoints saved in: checkpoints/")
print("="*80)