import numpy as np
import mlx.nn as nn
import mlx.core as mx
from utils import accuracy, save_checkpoint
from utils import CSVLogger
from models.vit_mlx import DeiTSmall
from data.data_loader import download_cifar100, load_cifar100, make_batches

def train(model, optimizer, loss_fn, train_data, train_labels, batch_size=64, epochs=10):
    logger = CSVLogger("training_log.csv")
    best_acc = 0.0

    for epoch in range(epochs):
        losses, accs = [], []
        for x, y in make_batches(train_data, train_labels, batch_size):
            def loss_fn_closure():
                logits = model(x)
                loss = loss_fn(logits, y)
                return loss, logits

            (loss, logits), grads = mx.grad(loss_fn_closure, model.parameters())
            optimizer.update(model.parameters(), grads)

            acc = accuracy(logits, y)
            losses.append(loss.item())
            accs.append(acc)

        avg_loss = np.mean(losses)
        avg_acc = np.mean(accs)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}  Accuracy: {avg_acc*100:.2f}%")

        logger.log(epoch + 1, avg_loss, avg_acc)

        # Save best checkpoint
        if avg_acc > best_acc:
            best_acc = avg_acc
            save_checkpoint(model, name="best_model.npz")

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, name=f"model_epoch{epoch+1}.npz")

if __name__ == "__main__":
    download_cifar100()
    train_x, train_y = load_cifar100("train")
    test_x, test_y = load_cifar100("test")

    model = DeiTSmall(num_classes=100)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = nn.Adam(model.parameters(), lr=3e-4)

    train(model, optimizer, loss_fn, train_x, train_y)
