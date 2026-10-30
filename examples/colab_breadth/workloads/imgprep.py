"""E053 (docs/research/0268): image preprocessing as in a training pipeline — 1,500 synthetic
RGB photos (768 x 768), each resized with Pillow to 512 x 512 and normalized into one float32
batch (4.7 GB), then per-channel statistics."""

import numpy as np
from common import done, sha
from PIL import Image

N, SRC, DST = 1500, 768, 512
rng = np.random.default_rng(0)
yy, xx = np.mgrid[0:SRC, 0:SRC].astype(np.float32) / SRC
batch = np.empty((N, DST, DST, 3), np.float32)
mean = np.array([0.485, 0.456, 0.406], np.float32)
std = np.array([0.229, 0.224, 0.225], np.float32)
for i in range(N):
    base = np.stack([np.sin(xx * (3 + i % 7)), np.cos(yy * (2 + i % 5)), xx * yy], -1)
    img = ((base + 1) * 120 + rng.integers(0, 15, (SRC, SRC, 3))).clip(0, 255).astype(np.uint8)
    small = np.asarray(Image.fromarray(img).resize((DST, DST), Image.BILINEAR), np.float32)
    batch[i] = (small / 255 - mean) / std
done(
    channel_mean=sha(batch.mean(axis=(0, 1, 2))),
    channel_std=sha(batch.std(axis=(0, 1, 2))),
    batch=sha(batch),
)
