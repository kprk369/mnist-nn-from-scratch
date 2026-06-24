import numpy as np


# AI-Assisted code:

def load_mnist_images(filepath):
    with open(filepath, 'rb') as f:
        f.read(4)
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        buffer = f.read()
        images = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        images = images.reshape(num_images, rows * cols).T

        return images / 255.0


def load_mnist_labels(filepath):
    with open(filepath, 'rb') as f:
        f.read(4)
        num_labels = int.from_bytes(f.read(4), 'big')
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
        return labels

# AI-Assisted code end.