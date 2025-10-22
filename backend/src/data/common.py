import numpy as np
import torch

def set_channel(*args, n_channels=3):
    """Convert image to specified number of channels"""
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            # Convert to grayscale
            img = np.expand_dims(np.mean(img, axis=2), 2)
        elif n_channels == 3 and c == 1:
            # Convert grayscale to RGB
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255):
    """Convert numpy array to PyTorch tensor"""
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]
