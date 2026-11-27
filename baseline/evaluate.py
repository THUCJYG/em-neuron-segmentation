import argparse
import json
from pathlib import Path
from data import PROJECT, read_split
from engine import load_checkpoint, evaluate
from utils import setup, write_json, environment


def main():
    parser = argparse.ArgumentParser(description="Full 512x512 inference and affinity F1 / instance F1@50 / F1@75")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=PROJECT / "data")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--decoder-config", type=Path, help="Optional decoder JSON chosen on validation ONLY")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Use a new, empty --out directory")
    setup(2026, args.threads)
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    decoder = checkpoint["config"]["decoder"]
    if args.decoder_config:
        decoder = json.loads(args.decoder_config.read_text())
    result = evaluate(model, read_split(args.data, args.split), args.device, decoder, args.out)
    result.update(split=args.split, checkpoint=str(args.checkpoint),
                  inference_views=getattr(model, "inference_views", 1),
                  checkpoint_step=checkpoint["step"], environment=environment())
    write_json(args.out / "metrics.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
