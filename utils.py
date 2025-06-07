import csv
import time
from pathlib import Path

class CSVLogger:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.start_time = time.time()

        if not self.filepath.exists():
            with open(self.filepath, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "loss", "accuracy", "time_sec"])

    def log(self, epoch, loss, accuracy):
        elapsed = time.time() - self.start_time
        with open(self.filepath, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{loss:.4f}", f"{accuracy:.4f}", f"{elapsed:.2f}"])
