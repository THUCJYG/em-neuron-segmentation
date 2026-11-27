"""Time complete CPU training steps. This is NOT a convergence benchmark."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from affinity import affinity_loss
from data import PROJECT, read_split, sample_batch
from model import SmallUNet
from utils import setup, environment, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--synthetic", action="store_true", help="Timing-only synthetic inputs, explicitly recorded")
    parser.add_argument("--out", type=Path, default=PROJECT / "runs" / "cpu_benchmark.json")
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0:
        parser.error("steps must be positive, warmup nonnegative")
    setup(2026, args.threads)
    rng = np.random.default_rng(2026)
    if args.synthetic:
        labels = np.indices((512, 512))[1] // 32 + 1
        samples = [{"image": rng.integers(0, 256, (512, 512), dtype=np.uint8), "labels": labels}]
    else:
        samples = read_split(args.data, "train")
    net = SmallUNet(args.base_channels)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    durations = []
    for i in range(args.warmup + args.steps):
        start = time.perf_counter()
        x, target, valid = sample_batch(samples, rng, args.patch_size, args.batch_size)
        optimizer.zero_grad(set_to_none=True)
        loss = affinity_loss(net(x), target, valid)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        optimizer.step()
        if i >= args.warmup:
            durations.append(time.perf_counter() - start)
    mean = float(np.mean(durations))
    result = {"data_kind": "synthetic timing only" if args.synthetic else "real CREMI training crops",
              "patch_size": args.patch_size, "batch_size": args.batch_size,
              "base_channels": args.base_channels, "parameters": sum(p.numel() for p in net.parameters()),
              "warmup_steps": args.warmup, "measured_steps": args.steps,
              "mean_step_seconds": mean, "p95_step_seconds": float(np.percentile(durations, 95)),
              "projected_5000_training_seconds": 5000 * mean,
              "projection_excludes": "validation, plotting, startup, checkpoint I/O; not a completed run",
              "environment": environment()}
    write_json(args.out, result)
    print(result)


if __name__ == "__main__":
    main()
