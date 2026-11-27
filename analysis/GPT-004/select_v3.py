"""Select a frozen network, inference transform count and watershed settings on B."""
import csv
import itertools
import sys
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "baseline"))
from affinity import boundary_counts, f1_score, make_affinities
from data import read_split
from engine import evaluate, load_checkpoint, predict
from metrics import InstanceF1
from postprocess import DEFAULT_DECODER, decode, merge_regions
from utils import setup, write_json

NETWORKS = ("gpt_v2_long", "gpt_v3_wide")
# Order also resolves exact ties: prefer default settings and single-view inference.
SETTINGS = tuple(itertools.product((.35, .20, .50), (16, 8, 32)))


def pool(groups):
    pooled = InstanceF1()
    for metric in groups:
        pooled.total_gt += metric.total_gt
        pooled.total_predictions += metric.total_predictions
        for t in pooled.thresholds:
            pooled.tp[t] += metric.tp[t]
    return pooled.compute()


def main():
    setup()
    final = ROOT / "runs/gpt_v3_selected"
    if final.exists() and any(final.iterdir()):
        raise ValueError("Use a fresh output directory for the selected checkpoint")
    validation = read_split(ROOT / "data", "val")
    records, by_slice = [], []
    for run, views in itertools.product(NETWORKS, (1, 8)):
        model, saved = load_checkpoint(ROOT / "runs" / run / "best.pt")
        model.eval()
        metrics = {}
        tp = fp = fn = 0
        for sample in validation:
            section = sample["name"].split("_")[1]
            prob = predict(model, sample["image"], views=views)
            target, valid = make_affinities(sample["labels"])
            a,b,c = boundary_counts(prob,target,valid)
            tp,fp,fn = tp+a,fp+b,fn+c
            for threshold,size in SETTINGS:
                pred = decode(prob, seed_threshold=threshold, min_seed_size=size,
                              smooth_sigma=DEFAULT_DECODER["smooth_sigma"])
                choices = (None,.55,.75) if threshold <= .35 and size <= 16 else (None,)
                for merge in choices:
                    labels = pred if merge is None else merge_regions(pred,prob,merge)
                    key = (threshold,size,merge)
                    if key not in metrics:
                        metrics[key] = {}
                    if section not in metrics[key]:
                        metrics[key][section] = InstanceF1()
                    metrics[key][section].update(sample["labels"],labels)
        for (threshold,size,merge), groups in metrics.items():
            scores = pool(groups.values())
            by_slice.append(groups)
            if views == 1 and (threshold,size,merge)==(.35,16,None):
                assert scores["F1_75"] == saved["validation"]["F1_75"]
            records.append(dict(network=run,step=saved["step"],views=views,
                                seed_threshold=threshold,min_seed_size=size,merge_affinity=merge,
                                affinity_f1=f1_score(tp,fp,fn),**scores))
        print("Validated",run,views,"views on",len(validation),"images",flush=True)
    with (OUT/"validation_grid.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    winner = max(records,key=lambda r:(r["F1_75"],r["F1_50"]))
    folds, heldout = [], []
    for section in sorted(by_slice[0]):
        others = [pool(m for z,m in groups.items() if z != section) for groups in by_slice]
        idx = max(range(len(records)),key=lambda i:(others[i]["F1_75"],others[i]["F1_50"]))
        heldout.append(by_slice[idx][section])
        folds.append(dict(section=section,settings={k:records[idx][k] for k in
            ("network","views","seed_threshold","min_seed_size","merge_affinity")},
            heldout=heldout[-1].compute(),v2=by_slice[0][section].compute()))
    write_json(OUT/"slice_check.json",dict(folds=folds,pooled=pool(heldout),v2=pool(by_slice[0].values())))
    decoder = dict(DEFAULT_DECODER,seed_threshold=winner["seed_threshold"],min_seed_size=winner["min_seed_size"])
    if winner["merge_affinity"] is not None:
        decoder["merge_affinity"] = winner["merge_affinity"]
    model, checkpoint = load_checkpoint(ROOT/"runs"/winner["network"]/"best.pt")
    model.inference_views = winner["views"]
    verified = evaluate(model,validation,decoder=decoder)
    for k in ("F1_75","F1_50","affinity_f1","TP_75","TP_50"):
        assert verified[k] == winner[k], k
    config=dict(checkpoint["config"],decoder=decoder,inference_views=winner["views"],
                source_network=winner["network"],analysis_id="GPT-004",network_update=3)
    final.mkdir(parents=True,exist_ok=True)
    torch.save(dict(checkpoint,config=config,validation=verified),final/"best.pt")
    write_json(final/"config.json",config)
    write_json(OUT/"selection.json",dict(source="GPT",analysis_update=4,network_update=3,
                                        criterion=["F1_75","F1_50"],winner=winner))
    print("PASS: grid winner reproduced by standard evaluator",winner,flush=True)


if __name__ == "__main__":
    main()
