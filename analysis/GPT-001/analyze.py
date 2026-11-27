"""Rebuild validation diagnostics from the frozen baseline evaluation."""
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "baseline"))
from engine import load_checkpoint, predict
from postprocess import boundary_surface, decode
from utils import color_labels, setup

EVAL = ROOT / "runs/baseline_5000/val"


def load_png(path):
    with Image.open(path) as image:
        return np.array(image)


def load_image(stem):
    prefix = EVAL / "predictions" / stem
    return (
        load_png(ROOT / "data/val/images" / (stem + ".png")),
        load_png(ROOT / "data/val/labels" / (stem + ".png")),
        load_png(str(prefix) + "_instances.png"),
        [load_png(str(prefix) + "_affinity_" + direction + ".png")
         for direction in ("right", "down")],
    )


def candidates(primary_ids, other_ids, overlap):
    """Return primary objects containing at least two substantial, pure parts."""
    primary_area = overlap.sum(axis=1)
    other_area = overlap.sum(axis=0)
    eligible = ((overlap >= 16) & (overlap >= .15 * primary_area[:, None])
                & (overlap >= .65 * other_area))
    for i in np.flatnonzero(eligible.sum(axis=1) >= 2):
        for j, k in combinations(np.flatnonzero(eligible[i]), 2):
            yield dict(primary=int(primary_ids[i]), parts=[int(other_ids[j]), int(other_ids[k])],
                       pixels=[int(overlap[i, j]), int(overlap[i, k])],
                       primary_area=int(primary_area[i]),
                       part_areas=[int(other_area[j]), int(other_area[k])])


def interface(primary, other, affinity, case):
    # Restrict both endpoints to the same primary object and the selected pair.
    ys, xs, values = [], [], []
    a, b = case["parts"]
    for direction, (left, right) in enumerate([
        (np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])
    ]):
        pair = (((other[left] == a) & (other[right] == b))
                | ((other[left] == b) & (other[right] == a)))
        y, x = np.where(pair & (primary[left] == case["primary"])
                        & (primary[right] == case["primary"]))
        ys.extend(y)
        xs.extend(x)
        values.extend(affinity[direction][y, x])
    return np.array(ys, dtype=int), np.array(xs, dtype=int), np.array(values) / 255.


def save_example(kind, case, model, decoder):
    em, gt, pred, affinity = load_image(case["image"])
    primary, parts = (gt, pred) if kind == "split" else (pred, gt)
    probability = predict(model, em)
    assert np.array_equal(decode(probability, **decoder), pred)
    surface = ndi.gaussian_filter(boundary_surface(probability), decoder["smooth_sigma"])
    labels, _ = ndi.label(surface < decoder["seed_threshold"])
    keep = np.bincount(labels.ravel()) >= decoder["min_seed_size"]
    keep[0] = False
    seeds, _ = ndi.label(keep[labels])
    seed_mask = seeds > 0
    assert np.array_equal(pred[seed_mask], seeds[seed_mask])
    case["seed_pixels_in_intersections"] = [
        int((seed_mask & (primary == case["primary"]) & (parts == part)).sum())
        for part in case["parts"]
    ]
    y, x, p = interface(primary, parts, affinity, case)
    yy, xx = np.where((primary == case["primary"]) & np.isin(parts, case["parts"]))
    y0, y1 = max(0, yy.min()-10), min(em.shape[0], yy.max()+11)
    x0, x1 = max(0, xx.min()-10), min(em.shape[1], xx.max()+11)
    roi = np.s_[y0:y1, x0:x1]
    case["crop_xyxy_exclusive"] = list(map(int, (x0, y0, x1, y1)))
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.9), layout="constrained")
    panels = ((em, "EM"), (color_labels(gt), "Ground truth"),
              (color_labels(pred), "Prediction"), (em, "Affinity on shared interface"))
    for ax, (image, title) in zip(axes, panels):
        ax.imshow(image[roi], cmap="gray", interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    for ax in axes[1:3]:
        ax.scatter(x-x0, y-y0, s=1, c="white", alpha=.75)
    dots = axes[3].scatter(x-x0, y-y0, c=p, s=7, cmap="coolwarm", vmin=0, vmax=1)
    fig.colorbar(dots, ax=axes[3], shrink=.7, label="P(same instance)")
    fig.suptitle(f"GPT-001  |  {kind}  |  {case['image']}  |  x={x0}:{x1}, y={y0}:{y1}")
    fig.savefig(OUT / (kind + ".png"), dpi=180)
    plt.close(fig)


def main():
    setup()
    model, checkpoint = load_checkpoint(EVAL.parent / "best.pt")
    model.eval()
    metrics = json.loads((EVAL / "metrics.json").read_text())
    rows = sorted(metrics["per_image"], key=lambda r: (r["F1_75"], r["F1_50"], r["image"]))
    best = {}
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        _, gt, pred, affinity = load_image(row["image"])
        valid = gt > 0
        gids, gi = np.unique(gt[valid], return_inverse=True)
        pids, pi = np.unique(pred[valid], return_inverse=True)
        overlap = np.bincount(gi * len(pids) + pi, minlength=len(gids)*len(pids)).reshape(len(gids), -1)
        areas = overlap.sum(axis=1)
        iou = overlap / (areas[:, None] + overlap.sum(axis=0) - overlap)
        row["gt_area_le64"] = int((areas <= 64).sum())
        row["small_no_iou50"] = int(((areas <= 64) & (iou.max(axis=1) < .5)).sum())
        for kind, primary, parts, ids, others, table in (
            ("split", gt, pred, gids, pids, overlap), ("merge", pred, gt, pids, gids, overlap.T)
        ):
            cases = list(candidates(ids, others, table))
            row[kind + "_candidates"] = len({c["primary"] for c in cases})
            if rank > 3:
                continue
            for c in cases:
                _, _, p = interface(primary, parts, affinity, c)
                if not len(p):
                    continue
                wrong = p < .5 if kind == "split" else p >= .5
                key = (int(wrong.sum()), len(p))
                if kind not in best or key > best[kind][0]:
                    c.update(image=row["image"], interface_edges=len(p),
                             affinity_errors=int(wrong.sum()), mean_p=float(p.mean()))
                    best[kind] = key, c
    with (OUT / "validation_ranking.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert set(best) == {"split", "merge"}
    cases = {kind: value[1] for kind, value in best.items()}
    for kind, case in cases.items():
        case["case_id"] = {"split": "GPT-001-S01", "merge": "GPT-001-M01"}[kind]
        save_example(kind, case, model, checkpoint["config"]["decoder"])
    result = dict(source="GPT", update=1, checkpoint_step=metrics["checkpoint_step"], cases=cases)
    (OUT / "cases.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
