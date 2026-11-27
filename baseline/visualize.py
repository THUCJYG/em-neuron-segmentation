"""Make readable GT examples, predictions and training curves with Matplotlib."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from affinity import make_affinities
from data import PROJECT, read_split
from engine import load_checkpoint, predict
from postprocess import decode
from utils import color_labels, setup


def examples(sample, output, model=None, decoder=None):
    target, valid = make_affinities(sample["labels"])
    rows = 2 if model is not None else 1
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3.7 * rows), squeeze=False)
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("#4285c6")
    panels = [(sample["image"], "EM image", "gray"),
              (np.ma.array(target[0], mask=~valid[0]), "GT right affinity", cmap),
              (np.ma.array(target[1], mask=~valid[1]), "GT down affinity", cmap),
              (color_labels(sample["labels"]), "GT 2D instances", None)]
    if model is not None:
        model.eval()
        probability = predict(model, sample["image"])
        instances = decode(probability, **decoder)
        panels += [(sample["image"], "EM image (same crop)", "gray"),
                   (probability[0], "Predicted right affinity", "gray"),
                   (probability[1], "Predicted down affinity", "gray"),
                   (color_labels(instances), "Predicted instances", None)]
    for ax, (values, title, color) in zip(axes.flat, panels):
        kwargs = {"vmin": 0, "vmax": 1} if "affinity" in title else {}
        ax.imshow(values, cmap=color, interpolation="nearest", **kwargs)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.text(.5, .012, f"{sample['name']} | affinity: white = 1, black = 0, blue = ignored | instance colors are arbitrary",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .97), pad=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=.1)
    plt.close(fig)


def curves(history_path, output):
    history = json.loads(history_path.read_text())
    train = [r for r in history if "train_loss" in r]
    val = [r for r in history if "validation" in r]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    axes[0].plot([r["step"] for r in train], [r["train_loss"] for r in train])
    axes[0].set_title("Training loss (100-step means)")
    axes[1].plot([r["step"] for r in train], [r["train_affinity_f1"] for r in train], label="train")
    axes[1].plot([r["step"] for r in val], [r["validation"]["affinity_f1"] for r in val], "o-", label="validation")
    axes[1].set_title("Affinity F1 (boundary edges)")
    for key in ("F1_50", "F1_75"):
        axes[2].plot([r["step"] for r in val], [r["validation"][key] for r in val], "o-", label=key.replace("_", "@"))
    axes[2].set_title("Validation instance F1")
    for ax in axes:
        ax.set_xlabel("Training iteration")
        ax.grid(alpha=.2)
    axes[1].legend(); axes[2].legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=PROJECT / "data")
    p.add_argument("--split", choices=("train", "val", "test"), default="train")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--history", type=Path)
    p.add_argument("--out", type=Path, default=PROJECT / "figures" / "gt_example.png")
    args = p.parse_args()
    setup()
    if args.history:
        curves(args.history, args.out)
    else:
        sample = read_split(args.data, args.split)[args.index]
        if args.checkpoint:
            model, saved = load_checkpoint(args.checkpoint)
            examples(sample, args.out, model, saved["config"]["decoder"])
        else:
            examples(sample, args.out)


if __name__ == "__main__":
    main()
