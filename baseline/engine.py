"""Full-resolution inference and fixed evaluation; no GT-dependent decoding."""
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from affinity import make_affinities, boundary_counts, f1_score
from data import normalize
from metrics import InstanceF1
from model import SmallUNet
from postprocess import decode, DEFAULT_DECODER
from utils import color_labels, write_json


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    model = SmallUNet(config["base_channels"], config.get("architecture", "small_unet"),
                      config.get("aux_weight", 0) > 0, 2 * len(config.get("long_distances", []))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.inference_views = config.get("inference_views", 1)
    return model, checkpoint


@torch.inference_mode()
def predict(model, image, device="cpu", views=None):
    tensor = torch.from_numpy(normalize(image)[None, None]).to(device)
    views = getattr(model, "inference_views", 1) if views is None else views
    if views == 1:
        return torch.sigmoid(model(tensor))[0].cpu().numpy()
    if views != 8:
        raise ValueError("Expected one or eight inference views")
    total = torch.zeros((2, *image.shape), device=device)
    for transpose in (False, True):
        base = tensor.transpose(-2, -1) if transpose else tensor
        for axes in ((), (-1,), (-2,), (-2, -1)):
            view = base.flip(axes) if axes else base
            prob = torch.sigmoid(model(view))[0]
            if axes:
                prob = prob.flip(axes)
            # A reflected right/down edge is anchored at its opposite endpoint.
            right, down = prob.unbind()
            if -1 in axes:
                right = right.roll(-1, dims=-1)
            if -2 in axes:
                down = down.roll(-1, dims=-2)
            aligned = torch.stack((down.T, right.T)) if transpose else torch.stack((right, down))
            total += aligned
    return (total / views).cpu().numpy()


@torch.inference_mode()
def evaluate(model, samples, device="cpu", decoder=None, output_dir=None):
    model.eval()
    decoder = dict(DEFAULT_DECODER if decoder is None else decoder)
    metric = InstanceF1()
    tp, fp, fn, total = 0, 0, 0, 0
    records = []
    start = time.perf_counter()
    for sample in samples:
        probability = predict(model, sample["image"], device)
        # Decoding receives probabilities ONLY. GT is used below for evaluation.
        instances = decode(probability, **decoder)
        target, valid = make_affinities(sample["labels"])
        counts = boundary_counts(probability, target, valid)
        tp, fp, fn = tp + counts[0], fp + counts[1], fn + counts[2]
        total += int(valid.sum())
        metric.update(sample["labels"], instances)
        per_image = InstanceF1()
        per_image.update(sample["labels"], instances)
        records.append({"image": sample["name"], "affinity_f1": f1_score(*counts),
                        **per_image.compute()})
        if output_dir is not None:
            if instances.max() > 65535:
                raise ValueError("Too many predicted IDs for a lossless uint16 PNG")
            folder = Path(output_dir) / "predictions"
            folder.mkdir(parents=True, exist_ok=True)
            stem = sample["name"]
            Image.fromarray(instances.astype(np.uint16)).save(folder / f"{stem}_instances.png")
            Image.fromarray(color_labels(instances)).save(folder / f"{stem}_color.png")
            for d, name in enumerate(("right", "down")):
                Image.fromarray(np.rint(probability[d] * 255).astype(np.uint8)).save(
                    folder / f"{stem}_affinity_{name}.png")
    # Affinity F1 pools boundary-edge counts over the whole split (not a mean of per-image F1).
    result = {"affinity_f1": f1_score(tp, fp, fn),
              "boundary_precision": tp / (tp + fp) if tp + fp else None,
              "boundary_recall": tp / (tp + fn) if tp + fn else None,
              "valid_edges": total, "boundary_edges": tp + fn,
              **metric.compute(), "images": len(samples), "decoder": decoder,
              "evaluation_seconds": time.perf_counter() - start, "per_image": records}
    if output_dir is not None:
        write_json(Path(output_dir) / "metrics.json", result)
    return result
