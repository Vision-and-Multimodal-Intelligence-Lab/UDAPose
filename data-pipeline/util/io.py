import os

import numpy as np
import torch
from torchvision.transforms.v2 import functional as F


def save_image(img, path):
    if isinstance(img, np.ndarray):
        img = F.to_image(img)

    if img.ndim == 4:
        assert img.shape[0] == 1 and img.shape[1] == 3, f"img.shape: {img.shape}"
        img = img[0]
    elif img.ndim == 3:
        assert img.shape[0] == 3, f"img.shape: {img.shape}"
    else:
        raise ValueError(f"Invalid image shape: {img.shape}")

    assert img.dtype == torch.uint8, f"img.dtype: {img.dtype}"

    img = F.to_pil_image(img.cpu())

    ext = os.path.splitext(path)[1]
    if ext == ".png":
        img.save(
            path,
        )
    elif ext == ".webp":
        img.save(
            path,
            lossless=True,
        )
    else:
        raise NotImplementedError(f"Unsupported image format: {ext}")
