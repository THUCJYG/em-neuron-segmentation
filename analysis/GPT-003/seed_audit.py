"""Inspect the same watershed markers used in inference, using GT only for auditing."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT.parent / "GPT-002"))
from compare import candidates, diagnostic_counts, load_checkpoint, load_png, read
from data import read_split
from engine import predict
from postprocess import boundary_surface, decode
from utils import setup

RUNS = {"v1": "gpt_v1_dice", "v2": "gpt_v2_long"}


def seed_maps(prob, decoder):
    surface = ndi.gaussian_filter(boundary_surface(prob), decoder["smooth_sigma"])
    raw, _ = ndi.label(surface < decoder["seed_threshold"])
    keep = np.bincount(raw.ravel()) >= decoder["min_seed_size"]
    keep[0] = False
    seeds, _ = ndi.label(keep[raw])
    if not seeds.any():
        seeds.flat[surface.argmin()] = 1
    return raw, seeds


def audit(version):
    folder = ROOT / "runs" / RUNS[version]
    model, saved = load_checkpoint(folder / "best.pt")
    model.eval()
    decoder = saved["config"]["decoder"]
    fixed = read(OUT.parent / "GPT-001/cases.json")["cases"]["merge"]
    merges, small, fixed_seed = [], [], {}
    gallery = []
    for sample in read_split(ROOT / "data", "val"):
        stem, gt = sample["name"], sample["labels"]
        prob = predict(model, sample["image"])
        raw, seeds = seed_maps(prob, decoder)
        pred = decode(prob, **decoder)
        assert np.array_equal(pred, load_png(folder / "val/predictions" / (stem + "_instances.png")))
        assert np.array_equal(pred[seeds > 0], seeds[seeds > 0])
        counts, gids, pids, iou = diagnostic_counts(gt, pred)
        areas = np.bincount(gt.ravel())
        raw_area = np.bincount(raw.ravel())
        for idx in np.flatnonzero((areas[gids] <= 64) & (iou.max(axis=1) < .5)):
            gid = int(gids[idx])
            ids, sizes = np.unique(raw[gt == gid], return_counts=True)
            own = ids[(ids > 0) & (sizes == raw_area[ids])]
            if len(own) and raw_area[own].max() >= decoder["min_seed_size"]:
                reason = "independent_survives"
            elif len(own):
                reason = "independent_filtered"
            elif np.any(ids > 0):
                reason = "shared_only"
            else:
                reason = "no_raw_seed"
            small.append(dict(image=stem, gt=gid, area=int(areas[gid]), category=reason))
        valid = gt > 0
        overlap = np.bincount(np.searchsorted(pids, pred[valid])*len(gids)
                              + np.searchsorted(gids, gt[valid]), minlength=len(pids)*len(gids)).reshape(len(pids), -1)
        groups = {}
        for c in candidates(pids, gids, overlap):
            groups.setdefault(c["primary"], set()).update(c["parts"])
        assert len(groups) == counts["merge"]
        for pid, group in groups.items():
            hits = {g: int(((gt == g) & (seeds == pid)).sum()) for g in sorted(group)}
            merges.append(dict(image=stem, pred=pid, seed_pixels=hits,
                               shared=sum(v > 0 for v in hits.values()) >= 2))
        if groups:
            # EM background, yellow GT interfaces, cyan surviving markers, red weak affinity.
            rgb = np.repeat((sample["image"]/255.)[:, :, None], 3, axis=2)
            for pid, group in groups.items():
                focus = (pred == pid) & np.isin(gt, list(group))
                marker = (seeds == pid) & focus
                rgb[marker] = .65*rgb[marker] + .35*np.array([0, 1, 1])
                for axis in (0, 1):
                    s, t = (np.s_[:-1,:], np.s_[1:,:]) if axis == 0 else (np.s_[:,:-1], np.s_[:,1:])
                    edge = focus[s] & focus[t] & (gt[s] != gt[t])
                    rgb[s][edge] = [1, .85, 0]
                    rgb[s][edge & (prob[1-axis][s] >= .5)] = [1, .15, .1]
            gallery.append((stem, rgb))
        if stem == fixed["image"]:
            for g in fixed["parts"]:
                region = gt == g
                pid = int(np.bincount(pred[region]).argmax())
                fixed_seed[g] = dict(dominant_pred=pid, seed_pixels=int(((seeds == pid) & region).sum()))
    for start in range(0, len(gallery), 8):
        items = gallery[start:start+8]
        fig, axes = plt.subplots((len(items)+3)//4, 4, figsize=(12, 3.1*((len(items)+3)//4)),
                                 squeeze=False, layout="constrained")
        for ax in axes.flat:
            ax.axis("off")
        for ax, (stem, rgb) in zip(axes.flat, items):
            ax.imshow(rgb)
            ax.set_title(stem, fontsize=9)
        fig.suptitle(f"{version} merge candidates | cyan: seed | GT contact: red if P(same) >= .5, yellow otherwise", fontsize=11)
        fig.savefig(OUT / f"{version}_seeds_{start//8+1}.png", dpi=140)
        plt.close(fig)
    result = dict(source="GPT", analysis_update=3, network_update=2, audited=version,
                  merge_candidates=len(merges), shared=sum(m["shared"] for m in merges),
                  small_miss_categories=dict(Counter(r["category"] for r in small)),
                  M01=fixed_seed, merges=merges, small_misses=small)
    (OUT / f"{version}_seeds.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(version, {k: result[k] for k in ("merge_candidates","shared","small_miss_categories","M01")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version", choices=RUNS)
    setup()
    audit(parser.parse_args().version)
