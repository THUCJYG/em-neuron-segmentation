import json
import os
from pathlib import Path
import platform
import random
import time
import numpy as np
import torch


def setup(seed=2026, threads=4):
    if threads < 1:
        raise ValueError("threads must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def environment():
    import scipy
    import PIL
    return {"platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count(),
            "torch_threads": torch.get_num_threads(), "python": platform.python_version(),
            "torch": str(torch.__version__), "numpy": np.__version__,
            "scipy": scipy.__version__, "pillow": PIL.__version__,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def color_labels(labels):
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(25)
    colors = rng.integers(45, 245, size=(int(labels.max()) + 1, 3), dtype=np.uint8)
    colors[0] = 0
    return colors[labels]
