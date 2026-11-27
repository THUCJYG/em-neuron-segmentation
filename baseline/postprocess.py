"""Affinity -> scalar boundary surface -> SciPy seeded watershed.

This is a simple, lossy affinity-to-boundary conversion, NOT Mutex Watershed.
Inference never uses GT labels or a GT foreground mask.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


DEFAULT_DECODER = {"seed_threshold": 0.35, "min_seed_size": 16, "smooth_sigma": 0.6}


def boundary_surface(affinities):
    """At each pixel, take maximum dissimilarity on available incident edges."""
    a = np.clip(np.asarray(affinities, dtype=np.float32), 0, 1)
    if a.ndim != 3 or a.shape[0] != 2:
        raise ValueError("Expected right/down affinities with shape (2,H,W)")
    h, w = a.shape[1:]
    boundary = np.zeros((h, w), np.float32)
    right, down = 1 - a[0, :, :-1], 1 - a[1, :-1, :]
    boundary[:, :-1] = np.maximum(boundary[:, :-1], right)
    boundary[:, 1:] = np.maximum(boundary[:, 1:], right)
    boundary[:-1, :] = np.maximum(boundary[:-1, :], down)
    boundary[1:, :] = np.maximum(boundary[1:, :], down)
    return boundary


def merge_regions(instances, affinities, threshold):
    """Join touching regions with >=8 contact edges and high mean P(same)."""
    if not 0 <= threshold <= 1:
        raise ValueError("merge_affinity must be between 0 and 1")
    n = int(instances.max()) + 1
    pairs, weights = [], []
    for ch, (src, dst) in enumerate(((np.s_[:, :-1], np.s_[:, 1:]),
                                     (np.s_[:-1, :], np.s_[1:, :]))):
        a, b = instances[src], instances[dst]
        mask = a != b
        low = np.minimum(a[mask], b[mask]).astype(np.int64)
        high = np.maximum(a[mask], b[mask])
        pairs.append(low * n + high)
        weights.append(affinities[ch][src][mask])
    unique, index, count = np.unique(np.concatenate(pairs), return_inverse=True, return_counts=True)
    total = np.bincount(index, weights=np.concatenate(weights))
    join = unique[(count >= 8) & (total >= threshold * count)]
    if not len(join):
        return instances
    graph = coo_matrix((np.ones(len(join)), (join // n, join % n)), shape=(n, n))
    _, components = connected_components(graph, directed=False)
    return components[instances].astype(np.int32)


def marker_labels(boundary, seed_threshold, min_seed_size, core_threshold=None,
                  min_core_size=64, rescue_core_size=0):
    broad, _ = ndi.label(boundary < seed_threshold)
    sizes = np.bincount(broad.ravel())
    keep = sizes >= min_seed_size
    keep[0] = False
    marked = keep[broad]
    if core_threshold is not None:
        if not 0 <= core_threshold < seed_threshold or min_core_size < 1 or rescue_core_size < 0:
            raise ValueError("Invalid inner core threshold or area")
        cores, _ = ndi.label(boundary < core_threshold)
        core_sizes = np.bincount(cores.ravel())
        parent = np.zeros(len(core_sizes), dtype=np.int32)
        inside = cores > 0
        parent[cores[inside]] = broad[inside]
        substantial = core_sizes >= min_core_size
        substantial[0] = False
        split = np.bincount(parent[substantial], minlength=len(sizes)) >= 2
        marked = np.where(split[broad], substantial[cores], marked)
        if rescue_core_size:
            rescue = (core_sizes >= rescue_core_size) & ~keep[parent]
            rescue[0] = False
            marked |= rescue[cores]
    seeds, _ = ndi.label(marked)
    if seeds.max() == 0:
        seeds.flat[int(np.argmin(boundary))] = 1
    return seeds


def decode(affinities, seed_threshold=0.35, min_seed_size=16, smooth_sigma=0.6,
           merge_affinity=None, core_threshold=None, min_core_size=64, rescue_core_size=0):
    if not 0 <= seed_threshold <= 1 or min_seed_size < 1 or smooth_sigma < 0:
        raise ValueError("Expected threshold in [0,1], min_seed_size >= 1 and sigma >= 0")
    if not np.all(np.isfinite(affinities)):
        raise ValueError("Affinity predictions must be finite")
    boundary = boundary_surface(affinities)
    if smooth_sigma > 0:
        boundary = ndi.gaussian_filter(boundary, smooth_sigma)
    seeds = marker_labels(boundary, seed_threshold, min_seed_size, core_threshold,
                          min_core_size, rescue_core_size)
    surface = np.rint(np.clip(boundary, 0, 1) * 65535).astype(np.uint16)
    instances = ndi.watershed_ift(surface, seeds.astype(np.int32))
    if merge_affinity is not None:
        return merge_regions(instances, affinities, merge_affinity)
    return instances.astype(np.int32)
