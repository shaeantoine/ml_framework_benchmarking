import os 
import mlx.core as mx

def accuracy(logits, labels):
    preds = mx.argmax(logits, axis=1)
    return (preds == labels).astype(mx.float32).mean().item()

def save_checkpoint(model, path="checkpoints", name="model.npz"): 
    os.makedirs(path, exist_ok=True)
    model.save_weights(os.path.join(path, name))
    print(f"[Checkpoint saved] → {os.path.join(path, name)}")