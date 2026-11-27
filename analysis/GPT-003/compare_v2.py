"""Compare the frozen v2 validation checkpoint to v1 at the original GPT-001 interfaces."""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "GPT-002"))
from compare import ROOT, collect, diagnostic_counts, interface, load_image, load_png, read
from utils import color_labels

RUNS = {"v1": "gpt_v1_dice", "v2": "gpt_v2_long"}


def pred_path(version, stem, channel, runs=RUNS):
    return ROOT / "runs" / runs[version] / "val/predictions" / f"{stem}_{channel}.png"


def revisit(kind, case, runs=RUNS, out=OUT):
    stem = case["image"]
    em, gt, baseline, _ = load_image(stem)
    main, parts = (gt, baseline) if kind == "split" else (baseline, gt)
    ids = [case["primary"]] if kind == "split" else case["parts"]
    result = {}
    panels = [em, color_labels(gt)]
    for version in runs:
        pred = load_png(pred_path(version, stem, "instances", runs))
        aff = [load_png(pred_path(version, stem, "affinity_" + d, runs)) for d in ("right", "down")]
        y, x, prob = interface(main, parts, aff, case)
        same = [np.pad(pred[:, :-1] == pred[:, 1:], ((0,0),(0,1))),
                np.pad(pred[:-1, :] == pred[1:, :], ((0,1),(0,0)))]
        _, _, labels = interface(main, parts, [s.astype(np.uint8)*255 for s in same], case)
        assert len(prob) == case["interface_edges"]
        _, gids, _, iou = diagnostic_counts(gt, pred)
        result[version] = dict(edges=len(prob), affinity_errors=int(((prob < .5) if kind == "split" else
                                                                  (prob >= .5)).sum()),
                               decoded_errors=int(((labels == 0) if kind == "split" else (labels == 1)).sum()),
                               best_iou={str(g): float(iou[gids == g].max()) for g in ids},
                               dominant_pred={str(g): int(np.bincount(pred[gt == g]).argmax()) for g in ids})
        panels.append(color_labels(pred))
    x0,y0,x1,y1 = case["crop_xyxy_exclusive"]
    roi = np.s_[y0:y1,x0:x1]
    fig, axes = plt.subplots(1,4,figsize=(12,3.8),layout="constrained")
    for ax, panel, title in zip(axes, panels, ("EM + original interface","Ground truth",*runs)):
        ax.imshow(panel[roi], cmap="gray", interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    axes[0].scatter(x-x0,y-y0,s=2,c="gold")
    for ax in axes[1:]:
        ax.contour(np.isin(gt[roi],ids),levels=[.5],colors="white",linewidths=.65)
    fig.suptitle(case["case_id"] + "  |  " + stem)
    fig.savefig(out / (kind+".png"),dpi=160)
    plt.close(fig)
    return dict(image=stem, case_id=case["case_id"], **result)


def main():
    runs, totals, paired = {}, {}, {}
    fig, axes = plt.subplots(2,2,figsize=(10,6.4),layout="constrained")
    shared = ("architecture","seed","steps","batch_size","patch_size","lr","loss","aux_weight",
              "dice_weight","decoder","no_augment","base_channels","device","threads","val_every","log_every")
    config = read(ROOT / "runs" / RUNS["v1"] / "config.json")
    for version, name in RUNS.items():
        current = read(ROOT / "runs" / name / "config.json")
        assert all(current[k] == config[k] for k in shared)
        runs[version], metrics, history = collect(name)
        if version == "v2":
            assert current["long_distances"] == [4,8] and current["long_weight"] == .3
        for ax,key,title in zip(axes.flat, ("train_loss","affinity_f1","F1_50","F1_75"),
                                ("Training objective (100-step mean)","Validation boundary F1",
                                 "Validation instance F1@50","Validation instance F1@75")):
            pts = [(r["step"],r.get("validation",r)[key]) for r in history if key in r.get("validation",r)]
            ax.plot(*zip(*pts),label=version,marker=None if key=="train_loss" else "o")
            ax.set(xlabel="Optimizer updates",ylabel="Loss" if key=="train_loss" else "F1",title=title)
            ax.grid(alpha=.2)
            ax.legend()
        totals[version] = {}
        for r in metrics["per_image"]:
            stem = r["image"]
            gt = load_png(ROOT / "data/val/labels" / (stem+".png"))
            pred = load_png(pred_path(version,stem,"instances"))
            counts,gids,_,iou = diagnostic_counts(gt,pred)
            rows,cols = linear_sum_assignment(np.where(iou >= .5,iou,0),maximize=True)
            matched = rows[iou[rows,cols] >= .5]
            assert len(matched) == r["TP_50"]
            counts["small_tp50"] = int((np.bincount(gt.ravel())[gids[matched]] <= 64).sum())
            for k,v in counts.items():
                totals[version][k] = totals[version].get(k,0)+v
            paired.setdefault(stem,dict(image=stem)).update({version+"_"+k:v for k,v in
                {**counts, **{key:r[key] for key in ("affinity_f1","F1_50","F1_75")}}.items()})
        audit = read(OUT / (version+"_seeds.json"))
        assert totals[version]["merge"] == audit["merge_candidates"]
        assert totals[version]["small_unmatched"] == sum(audit["small_miss_categories"].values())
    fig.savefig(OUT / "learning_curves.png", dpi=165)
    plt.close(fig)
    for r in paired.values():
        r["delta_F1_75"] = r["v2_F1_75"] - r["v1_F1_75"]
    ordered = sorted(paired.values(),key=lambda r:r["delta_F1_75"])
    with (OUT/"per_image.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)
    cases={k:revisit(k,c) for k,c in read(OUT.parent/"GPT-001/cases.json")["cases"].items()}
    prior=read(OUT.parent/"GPT-002/comparison.json")
    assert runs["v1"]["weights_sha256"] == prior["runs"][RUNS["v1"]]["weights_sha256"]
    for kind,case in cases.items():
        assert case["v1"]["decoded_errors"] == prior["cases"][kind]["update_1"]["wrong_after_watershed"]
    best=max(runs,key=lambda v:(runs[v]["F1_75"],runs[v]["F1_50"]))
    result=dict(source="GPT",analysis_update=3,network_update=2,selected=best,runs=runs,
                counts=totals,cases=cases,improved=sum(r["delta_F1_75"]>0 for r in ordered),
                tied=sum(r["delta_F1_75"]==0 for r in ordered))
    (OUT/"comparison.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("PASS: matched configs, checkpoint reloads, 24 paired validation images and fixed interfaces")
    print("Selected",best,"F1@75",runs[best]["F1_75"])


if __name__ == "__main__":
    main()
