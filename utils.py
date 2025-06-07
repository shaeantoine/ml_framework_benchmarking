import mlx.core as mx

def accuracy(logits, labels):
    preds = mx.argmax(logits, axis=1)
    return (preds == labels).astype(mx.float32).mean().item()
