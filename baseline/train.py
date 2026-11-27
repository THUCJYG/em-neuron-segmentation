import argparse
from pathlib import Path
import time
import numpy as np
import torch
from affinity import affinity_loss, boundary_dice_loss, coarse_boundary_loss, f1_score
from data import PROJECT, read_split, sample_batch
from engine import evaluate
from model import SmallUNet
from postprocess import DEFAULT_DECODER
from utils import setup, write_json, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data")
    parser.add_argument("--out", type=Path, default=PROJECT / "runs" / "baseline")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--architecture", choices=("small_unet", "residual_unet"), default="small_unet")
    parser.add_argument("--aux-weight", type=float, default=0.0,
                        help="Weight of quarter-resolution boundary supervision")
    parser.add_argument("--dice-weight", type=float, default=0.0,
                        help="Soft Dice loss weight on valid right/down boundary edges")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--long-weight", type=float, default=0.,
                        help="Training-only right/down endpoint affinity loss at distances 4 and 8")
    parser.add_argument("--loss", choices=("balanced_bce", "bce"), default="balanced_bce")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.long_weight) or args.long_weight < 0:
        parser.error("--long-weight must be finite and nonnegative")
    if not np.isfinite(args.aux_weight) or args.aux_weight < 0:
        parser.error("--aux-weight must be finite and nonnegative")
    if not np.isfinite(args.dice_weight) or args.dice_weight < 0:
        parser.error("--dice-weight must be finite and nonnegative")
    if min(args.steps, args.batch_size, args.val_every, args.log_every, args.base_channels) < 1:
        parser.error("Steps, batch size, channel count, and intervals must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output directory is not empty. Use a NEW --out for each experiment.")
    setup(args.seed, args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["decoder"] = dict(DEFAULT_DECODER)
    long_distances = (4, 8) if args.long_weight else ()
    config["long_distances"] = list(long_distances)
    config["environment"] = environment()
    write_json(args.out / "config.json", config)
    training, validation = read_split(args.data, "train"), read_split(args.data, "val")
    model = SmallUNet(args.base_channels, args.architecture, args.aux_weight > 0,
                      2 * len(long_distances)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    history, best = [], (-1.0, -1.0)
    total_loss, tp, fp, fn, training_seconds = 0.0, 0, 0, 0, 0.0
    wall_start = time.perf_counter()
    print(f"parameters={sum(p.numel() for p in model.parameters()):,}; train={len(training)}, val={len(validation)}", flush=True)
    for step in range(1, args.steps + 1):
        model.train()
        t = time.perf_counter()
        x, target, valid = (v.to(args.device) for v in sample_batch(
            training, rng, args.patch_size, args.batch_size, not args.no_augment, (1, *long_distances)))
        optimizer.zero_grad(set_to_none=True)
        if long_distances:
            logits, coarse, long_logits = model(x, with_auxiliary=bool(args.aux_weight), with_long=True)
            long_loss = affinity_loss(long_logits, target[:, 2:], valid[:, 2:])
            target, valid = target[:, :2], valid[:, :2]
        elif args.aux_weight:
            logits, coarse = model(x, with_auxiliary=True)
        else:
            logits = model(x)
        loss = affinity_loss(logits, target, valid, args.loss)
        if long_distances:
            loss = loss + args.long_weight * long_loss
        if args.aux_weight:
            loss = loss + args.aux_weight * coarse_boundary_loss(coarse, target, valid)
        if args.dice_weight:
            loss = loss + args.dice_weight * boundary_dice_loss(logits, target, valid)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        with torch.no_grad():
            # Boundary edges (target 0) are the positive class; logit < 0 means p < 0.5.
            mask = valid > 0
            pred, truth = (logits < 0) & mask, (target < 0.5) & mask
            tp += int((pred & truth).sum())
            fp += int((pred & ~truth).sum())
            fn += int((~pred & truth).sum())
        total_loss += float(loss.detach())
        if args.device == "cuda":
            torch.cuda.synchronize()
        training_seconds += time.perf_counter() - t
        if step % args.log_every == 0 or step == args.steps:
            steps_in_window = step % args.log_every or args.log_every
            record = {"step": step, "train_loss": total_loss / steps_in_window,
                      "train_affinity_f1": f1_score(tp, fp, fn),
                      "training_seconds": training_seconds,
                      "wall_seconds": time.perf_counter() - wall_start}
            history.append(record)
            print(f"step={step}/{args.steps} loss={record['train_loss']:.4f} "
                  f"aff_f1={record['train_affinity_f1'] or 0.0:.4f} "
                  f"train_time={training_seconds:.1f}s", flush=True)
            total_loss, tp, fp, fn = 0.0, 0, 0, 0
        if step % args.val_every == 0 or step == args.steps:
            result = evaluate(model, validation, args.device, DEFAULT_DECODER)
            score = (result["F1_75"] or 0.0, result["F1_50"] or 0.0)
            record = {"step": step, "validation": result}
            history.append(record)
            payload = {"model": model.state_dict(), "config": config,
                       "step": step, "validation": result}
            torch.save(payload, args.out / "last.pt")
            if score > best:
                best = score
                torch.save(payload, args.out / "best.pt")
            print(f"validation: affinity_f1={result['affinity_f1'] or 0.0:.4f} "
                  f"F1_50={result['F1_50'] or 0.0:.4f} F1_75={result['F1_75'] or 0.0:.4f}", flush=True)
            write_json(args.out / "history.json", history)
    summary = {"steps_completed": args.steps, "training_seconds": training_seconds,
               "wall_seconds": time.perf_counter() - wall_start,
               "best_validation_F1_75": best[0], "best_validation_F1_50": best[1],
               "under_two_hours": time.perf_counter() - wall_start < 7200,
               "environment": environment()}
    write_json(args.out / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
