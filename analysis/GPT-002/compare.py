"""Verify the 2x2 ablation and revisit the exact GPT-001 validation interfaces."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "baseline"))
sys.path.insert(0, str(OUT.parent / "GPT-001"))
from analyze import candidates, interface, load_image, load_png
from engine import load_checkpoint
from utils import color_labels

NAMES = {"baseline_5000": "Baseline", "gpt_v1_residual": "Residual",
         "gpt_v1_aux": "Auxiliary", "gpt_v1_full": "Residual + auxiliary",
         "gpt_v1_dice": "Residual + auxiliary + Dice"}
KEYS = ("affinity_f1", "boundary_precision", "boundary_recall", "F1_50",
        "precision_50", "recall_50", "F1_75", "precision_75", "recall_75",
        "gt_instances", "predicted_instances", "TP_50", "TP_75")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect(run):
    folder = ROOT / "runs" / run
    metrics = read(folder / "val/metrics.json")
    history = read(folder / "history.json")
    val = [r for r in history if "validation" in r]
    best = max(val, key=lambda r: (r["validation"]["F1_75"], r["validation"]["F1_50"]))
    model, ckpt = load_checkpoint(folder / "best.pt")
    summary = read(folder / "summary.json")
    assert [r["step"] for r in val] == [1000, 2000, 3000, 4000, 5000]
    assert ckpt["step"] == best["step"] == metrics["checkpoint_step"]
    assert metrics["per_image"] == best["validation"]["per_image"]
    assert all(metrics[k] == best["validation"][k] for k in KEYS)
    assert summary["steps_completed"] == 5000 and metrics["split"] == "val" and metrics["images"] == 24
    return dict(step=ckpt["step"], parameters=sum(p.numel() for p in model.parameters()),
                weights_sha256=hashlib.sha256((folder / "best.pt").read_bytes()).hexdigest(),
                training_seconds=summary["training_seconds"], wall_seconds=summary["wall_seconds"],
                **{k: metrics[k] for k in KEYS}), metrics, history


def plot_histories(histories):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for run, history in histories.items():
        for ax, field, title in zip(axes.flat, ["train_loss", "affinity_f1", "F1_50", "F1_75"],
                                    ["Training objective (100-step mean)", "Validation boundary F1",
                                     "Validation instance F1@50", "Validation instance F1@75"]):
            train = field == "train_loss"
            points = [(r["step"], r[field] if train else r["validation"][field])
                      for r in history if (field if train else "validation") in r]
            x, y = zip(*points)
            ax.plot(x, y, label=NAMES[run], marker=None if train else "o", lw=1.6)
            ax.set(title=title, xlabel="Optimizer updates", ylabel="Loss" if train else "F1")
            ax.grid(alpha=.18)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("GPT-002  |  CREMI A training and B validation")
    fig.savefig(OUT / "learning_curves.png", dpi=160)
    plt.close(fig)


def diagnostic_counts(gt, pred):
    valid = gt > 0
    gids, gi = np.unique(gt[valid], return_inverse=True)
    pids, pi = np.unique(pred[valid], return_inverse=True)
    overlap = np.bincount(gi * len(pids) + pi, minlength=len(gids)*len(pids)).reshape(len(gids), -1)
    area = overlap.sum(axis=1)
    iou = overlap / (area[:, None] + overlap.sum(axis=0) - overlap)
    counts = dict(split=len({c["primary"] for c in candidates(gids, pids, overlap)}),
                  merge=len({c["primary"] for c in candidates(pids, gids, overlap.T)}),
                  small=int((area <= 64).sum()),
                  small_unmatched=int(((area <= 64) & (iou.max(axis=1) < .5)).sum()))
    return counts, gids, pids, iou


def fixed_case(kind, case, run):
    stem = case["image"]
    em, gt, old, old_aff = load_image(stem)
    prefix = ROOT / "runs" / run / "val/predictions" / stem
    new = load_png(str(prefix) + "_instances.png")
    new_aff = [load_png(str(prefix) + "_affinity_" + d + ".png") for d in ("right", "down")]
    primary, parts = (gt, old) if kind == "split" else (old, gt)
    gt_ids = [case["primary"]] if kind == "split" else case["parts"]
    x0, y0, x1, y1 = case["crop_xyxy_exclusive"]
    roi = np.s_[y0:y1, x0:x1]
    result = dict(case_id=case["case_id"], image=stem, crop_xyxy_exclusive=case["crop_xyxy_exclusive"])
    for label, pred, affinity in (("baseline", old, old_aff), ("update_1", new, new_aff)):
        y, x, probs = interface(primary, parts, affinity, case)
        assert len(probs) == case["interface_edges"]
        wrong = probs < .5 if kind == "split" else probs >= .5
        same = [np.pad(pred[:, :-1] == pred[:, 1:], ((0, 0), (0, 1))),
                np.pad(pred[:-1, :] == pred[1:, :], ((0, 1), (0, 0)))]
        _, _, decoded = interface(primary, parts, [v.astype(np.uint8) * 255 for v in same], case)
        wrong_decoded = decoded == 0 if kind == "split" else decoded == 1
        _, gids, pids, iou = diagnostic_counts(gt, pred)
        objects = {}
        for gid in gt_ids:
            row = iou[gids == gid][0]
            ids, pixels = np.unique(pred[gt == gid], return_counts=True)
            objects[gid] = dict(best_iou=float(row.max()), best_iou_pred=int(pids[row.argmax()]),
                                dominant_pred=int(ids[pixels.argmax()]), coverage=float(pixels.max()/pixels.sum()))
        result[label] = dict(edges=len(probs), wrong=int(wrong.sum()), mean_affinity=float(probs.mean()),
                             wrong_after_watershed=int(wrong_decoded.sum()),
                             objects=objects)
    assert result["baseline"]["wrong"] == case["affinity_errors"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.9), layout="constrained")
    for ax, panel, title in zip(axes, [em, color_labels(gt), color_labels(old), color_labels(new)],
                                ["EM + reference interface", "Ground truth", "Baseline", NAMES[run]]):
        ax.imshow(panel[roi], cmap="gray", interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    axes[0].scatter(x-x0, y-y0, s=2, c="gold")
    for ax in axes[1:]:
        ax.contour(np.isin(gt[roi], gt_ids), levels=[.5], colors="white", linewidths=.6)
    fig.suptitle(f"{case['case_id']}  |  {stem}  |  x={x0}:{x1}, y={y0}:{y1}")
    fig.savefig(OUT / (kind + ".png"), dpi=180)
    plt.close(fig)
    return result


def main():
    runs, metrics, histories = {}, {}, {}
    base_config = read(ROOT / "runs/baseline_5000/config.json")
    shared = ("steps", "batch_size", "patch_size", "base_channels", "seed", "lr", "loss",
              "device", "threads", "val_every", "log_every", "no_augment", "decoder")
    for run in NAMES:
        runs[run], metrics[run], histories[run] = collect(run)
        config = read(ROOT / "runs" / run / "config.json")
        assert all(config[k] == base_config[k] for k in shared), run
    assert runs["baseline_5000"]["weights_sha256"] == "414c0a7d18eeb5460747ebaf5b1d574a38621c4d6c1bd579f58204ddf17e5787"
    selected = max(runs, key=lambda r: (runs[r]["F1_75"], runs[r]["F1_50"]))
    plot_histories(histories)
    baseline = {r["image"]: r for r in metrics["baseline_5000"]["per_image"]}
    rows = []
    for r in metrics[selected]["per_image"]:
        stem = r["image"]
        gt = load_png(ROOT / "data/val/labels" / (stem + ".png"))
        pred = load_png(ROOT / "runs" / selected / "val/predictions" / (stem + "_instances.png"))
        counts, *_ = diagnostic_counts(gt, pred)
        row = dict(image=stem, **counts)
        for key in ("affinity_f1", "F1_50", "F1_75"):
            row["baseline_" + key] = baseline[stem][key]
            row["update_" + key] = r[key]
            row["delta_" + key] = r[key] - baseline[stem][key]
        rows.append(row)
    rows.sort(key=lambda r: r["update_F1_75"])
    with (OUT / "per_image.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    old = read(OUT.parent / "GPT-001/cases.json")["cases"]
    cases = {kind: fixed_case(kind, case, selected) for kind, case in old.items()}
    result = dict(source="GPT", analysis_update=2, network_update=1, selected_run=selected,
                  runs=runs, diagnostics={k: sum(r[k] for r in rows) for k in counts},
                  improved_images_F1_75=sum(r["delta_F1_75"] > 0 for r in rows), cases=cases)
    (OUT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("PASS: 5 x 5000 updates, checkpoint reloads, full validation metrics, fixed interfaces")
    print("Selected", selected, "F1@75", runs[selected]["F1_75"])


if __name__ == "__main__":
    main()
