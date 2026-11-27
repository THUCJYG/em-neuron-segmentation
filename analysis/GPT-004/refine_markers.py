"""Check two-level watershed markers on the first-stage validation winner."""
import itertools
import sys
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent/"GPT-002"))
from compare import ROOT, diagnostic_counts, read
from data import read_split
from engine import load_checkpoint, predict
from metrics import InstanceF1
from postprocess import decode
from utils import setup, write_json


def main():
    setup()
    target = ROOT/"runs/gpt_v3_final"
    if target.exists() and any(target.iterdir()):
        raise ValueError("Use a new output directory")
    source = ROOT/"runs/gpt_v3_selected"
    model, saved = load_checkpoint(source/"best.pt")
    model.eval()
    base = saved["config"]["decoder"]
    configs = [base] + [dict(base,core_threshold=t,min_core_size=n,rescue_core_size=r)
                        for t,n,r in itertools.product((.10,.20),(64,256),(0,4))
                        if t < base["seed_threshold"]]
    metrics = [InstanceF1() for _ in configs]
    counts = [dict(split=0,merge=0,small_unmatched=0) for _ in configs]
    for sample in read_split(ROOT/"data","val"):
        prob = predict(model,sample["image"])
        for cfg,metric,totals in zip(configs,metrics,counts):
            pred = decode(prob,**cfg)
            metric.update(sample["labels"],pred)
            diagnostics,*_ = diagnostic_counts(sample["labels"],pred)
            for k in totals:
                totals[k] += diagnostics[k]
    rows = [dict(decoder=cfg,**m.compute(),**cnt) for cfg,m,cnt in zip(configs,metrics,counts)]
    assert rows[0]["F1_75"] == saved["validation"]["F1_75"]
    winner = max(rows,key=lambda r:(r["F1_75"],r["F1_50"]))
    target.mkdir(parents=True,exist_ok=True)
    config=dict(saved["config"],decoder=winner["decoder"])
    # The release carries the matching full evaluation, including per-image scores.
    from engine import evaluate
    verified=evaluate(model,read_split(ROOT/"data","val"),decoder=winner["decoder"])
    assert all(verified[k]==winner[k] for k in ("F1_75","F1_50","TP_75","TP_50"))
    torch.save(dict(saved,config=config,validation=verified),target/"best.pt")
    write_json(target/"config.json",config)
    write_json(OUT/"marker_comparison.json",dict(source="GPT",analysis_update=4,network_update=3,
                                                 rows=rows,winner=winner))
    print("PASS: marker comparison and full-precision evaluation", winner, flush=True)


if __name__=="__main__":
    main()
