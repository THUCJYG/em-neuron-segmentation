"""Save a decoder override for the lower-merge, small-core rescue variant."""
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent/"GPT-003"))
from compare_v2 import ROOT, read
from engine import load_checkpoint, predict
from postprocess import decode
from utils import setup, write_json
from compare import load_png
import numpy as np


def main():
    setup()
    rows=read(OUT/"marker_comparison.json")["rows"]
    v2=read(OUT.parent/"GPT-003/comparison.json")
    qualified=[r for r in rows if r["merge"]<=v2["counts"]["v2"]["merge"]
               and r["F1_75"]>v2["runs"]["v2"]["F1_75"]]
    if not qualified:
        return
    chosen=max(qualified,key=lambda r:(-r["small_unmatched"],r["F1_75"]))
    write_json(OUT/"lower_merge_decoder.json",chosen["decoder"])
    model,_=load_checkpoint(ROOT/"runs/gpt_v3_final/best.pt")
    model.eval()
    original=read(OUT.parent/"GPT-001/cases.json")["cases"]["merge"]
    stem=original["image"]
    gt=load_png(ROOT/"data/val/labels"/(stem+".png"))
    pred=decode(predict(model,load_png(ROOT/"data/val/images"/(stem+".png"))),**chosen["decoder"])
    dominant={str(g):int(np.bincount(pred[gt==g]).argmax()) for g in original["parts"]}
    write_json(OUT/"lower_merge_result.json",dict(source="GPT",analysis_update=4,network_update=3,
                                                  validation=chosen,M01_dominant_predictions=dominant))
    print("Lower-merge variant:",chosen["F1_75"],"M01:",dominant)


if __name__=="__main__":
    main()
