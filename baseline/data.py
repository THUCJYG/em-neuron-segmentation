"""Load ordinary PNG images and losslessly stored 16-bit instance-ID PNGs."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch

PROJECT = Path(__file__).resolve().parents[1]


def read_split(data_dir, split):
    root = Path(data_dir) / split
    paths = sorted((root / "images").glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG images in {root / 'images'}")
    samples = []
    for path in paths:
        with Image.open(path) as im:
            image = np.array(im.convert("L"), dtype=np.uint8)
        with Image.open(root / "labels" / path.name) as im:
            labels = np.array(im, dtype=np.int32)
        if image.shape != labels.shape or image.ndim != 2:
            raise ValueError(f"Image/label shape mismatch: {path.name}")
        samples.append({"name": path.stem, "image": image, "labels": labels})
    return samples


def normalize(image):
    # Fixed intensity scaling: does not learn statistics from validation/test.
    return (np.asarray(image, dtype=np.float32) / 255.0 - 0.5) / 0.5


def sample_batch(samples, rng, patch_size=128, batch_size=1, augment=True, distances=(1,)):
    from affinity import make_affinities
    images, targets, masks = [], [], []
    for _ in range(batch_size):
        sample = samples[int(rng.integers(len(samples)))]
        image, labels = sample["image"], sample["labels"]
        h, w = image.shape
        if patch_size > min(h, w) or patch_size < 8:
            raise ValueError("patch_size must be between 8 and the image size")
        y = int(rng.integers(h - patch_size + 1))
        x = int(rng.integers(w - patch_size + 1))
        image = image[y:y + patch_size, x:x + patch_size]
        labels = labels[y:y + patch_size, x:x + patch_size]
        if augment:
            k = int(rng.integers(4))
            image, labels = np.rot90(image, k), np.rot90(labels, k)
            if rng.random() < 0.5:
                image, labels = image[:, ::-1], labels[:, ::-1]
        # Regenerate targets after geometry transforms: directions stay correct.
        pairs = [make_affinities(labels, d) for d in distances]
        if len(pairs) == 1:
            target, valid = pairs[0]
        else:
            target, valid = (np.concatenate(values) for values in zip(*pairs))
        images.append(normalize(image)[None])
        targets.append(target)
        masks.append(valid)
    return tuple(torch.from_numpy(np.stack(arrays)) for arrays in (images, targets, masks))
