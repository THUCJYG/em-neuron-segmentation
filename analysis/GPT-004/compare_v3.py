"""Verify the release checkpoint, plot training curves and revisit fixed error cases."""
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent/"GPT-003"))
from compare_v2 import ROOT, collect, diagnostic_counts, load_png, plt, read, revisit
from scipy import ndimage as ndi
from postprocess import boundary_surface, marker_labels
from engine import load_checkpoint, predict
from utils import setup, write_json

RUNS = {"v2":"gpt_v2_long", "v3":"gpt_v3_final"}


def main():
    setup()
    runs = ROOT/"runs"
    final = runs/RUNS["v3"]
    model, ckpt = load_checkpoint(final/"best.pt")
    model.eval()
    chosen = read(OUT/"selection.json")["winner"]
    marker = read(OUT/"marker_comparison.json")["winner"]
    metrics = {v:read(runs/name/"val/metrics.json") for v,name in RUNS.items()}
    assert metrics["v3"]["per_image"] == ckpt["validation"]["per_image"]
    assert metrics["v3"]["F1_75"] == marker["F1_75"]
    assert metrics["v3"]["decoder"] == ckpt["config"]["decoder"]
    assert metrics["v3"]["checkpoint_step"] == chosen["step"]
    nets = {}
    fig, axes = plt.subplots(2,2,figsize=(10,7),layout="constrained")
    for run in ("gpt_v2_long","gpt_v3_wide"):
        nets[run], _, history = collect(run)
        for ax,key,title in zip(axes.flat,("train_loss","affinity_f1","F1_50","F1_75"),
                                ("Training loss (100-step mean)","Validation boundary F1",
                                 "Validation instance F1@50","Validation instance F1@75")):
            pts=[(r["step"],r.get("validation",r)[key]) for r in history if key in r.get("validation",r)]
            ax.plot(*zip(*pts), label="8 base channels" if run=="gpt_v2_long" else "16 base channels")
            ax.set(xlabel="Optimizer updates",ylabel="Loss" if key=="train_loss" else "F1",title=title)
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
    fig.suptitle("Training comparison | default watershed, one view")
    fig.savefig(OUT/"learning_curves.png",dpi=160)
    plt.close(fig)
    totals, paired = {}, {}
    for v, name in RUNS.items():
        totals[v]={}
        for r in metrics[v]["per_image"]:
            stem=r["image"]
            gt=load_png(ROOT/"data/val/labels"/(stem+".png"))
            pred=load_png(runs/name/"val/predictions"/(stem+"_instances.png"))
            counts,gids,_,iou=diagnostic_counts(gt,pred)
            rows,cols=linear_sum_assignment(np.where(iou>=.5,iou,0),maximize=True)
            matched=rows[iou[rows,cols]>=.5]
            assert len(matched)==r["TP_50"]
            counts["small_tp50"]=int((np.bincount(gt.ravel())[gids[matched]]<=64).sum())
            for k,n in counts.items():
                totals[v][k]=totals[v].get(k,0)+n
            paired.setdefault(stem,{"image":stem}).update({f"{v}_{k}":n for k,n in
                {**counts,**{k:r[k] for k in ("affinity_f1","F1_50","F1_75")}}.items()})
    for row in paired.values():
        row["delta_F1_75"]=row["v3_F1_75"]-row["v2_F1_75"]
    ordered=sorted(paired.values(),key=lambda r:r["delta_F1_75"])
    with (OUT/"per_image.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)
    originals=read(OUT.parent/"GPT-001/cases.json")["cases"]
    cases={k:revisit(k,c,RUNS,OUT) for k,c in originals.items()}
    prior=read(OUT.parent/"GPT-003/comparison.json")
    assert totals["v2"]==prior["counts"]["v2"]
    for kind in cases:
        assert cases[kind]["v2"]==prior["cases"][kind]["v2"]
    fixed=originals["merge"]
    img=load_png(ROOT/"data/val/images"/(fixed["image"]+".png"))
    gt=load_png(ROOT/"data/val/labels"/(fixed["image"]+".png"))
    pred=load_png(final/"val/predictions"/(fixed["image"]+"_instances.png"))
    decoder=ckpt["config"]["decoder"]
    surface=ndi.gaussian_filter(boundary_surface(predict(model,img)),decoder["smooth_sigma"])
    seeds=marker_labels(surface,**{k:v for k,v in decoder.items() if k not in ("smooth_sigma","merge_affinity")})
    # Record all pre-merge seeds that contribute to the dominant final region in each GT.
    seed_hits={}
    for gid in fixed["parts"]:
        pid=cases["merge"]["v3"]["dominant_pred"][str(gid)]
        region=(gt==gid)&(pred==pid)&(seeds>0)
        ids,counts=np.unique(seeds[region],return_counts=True)
        seed_hits[str(gid)]={str(i):int(n) for i,n in zip(ids,counts)}
    result=dict(source="GPT",analysis_update=4,network_update=3,stage_one=chosen,chosen=marker,networks=nets,
                counts=totals,cases=cases,M01_seed_intersections=seed_hits,
                improved=sum(r["delta_F1_75"]>0 for r in ordered),tied=sum(r["delta_F1_75"]==0 for r in ordered),
                release_sha256=hashlib.sha256((final/"best.pt").read_bytes()).hexdigest())
    write_json(OUT/"comparison.json",result)
    print("PASS: released checkpoint, 24 official per-image records, matched diagnostics and original cases")


if __name__=="__main__":
    main()
