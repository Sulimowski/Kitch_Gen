import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .condition import SCALE, condition_vector, network_inputs
from .shapes import KINDS, random_params, shape_bbox, shape_sdf


def surface_samples(fn, lo, hi, n, rng, budget=300_000):
    h = max(0.002, (float(np.prod(hi - lo)) / budget) ** (1 / 3))
    axes = [np.arange(lo[i], hi[i] + h, h) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    band = grid[np.abs(fn(grid)) < h]
    p = band[rng.integers(0, len(band), n)] + rng.uniform(-h / 2, h / 2, (n, 3))
    eps = 1e-4
    for _ in range(2):
        d = fn(p)
        grad = np.stack([(fn(p + e) - fn(p - e)) / (2 * eps) for e in np.eye(3) * eps], -1)
        p = p - d[:, None] * grad / np.maximum(np.sum(grad * grad, 1, keepdims=True), 1e-12)
    return p


def _build_chunk(task):
    start, stop, points_per_shape, seed = task
    rng = np.random.default_rng(seed)
    n_near = int(points_per_shape * 0.85)
    pts_all, sdf_all, idx_all, conds = [], [], [], []
    for s in range(start, stop):
        kind = KINDS[s % len(KINDS)]
        P = random_params(kind, rng)
        lo, hi = shape_bbox(kind, P)
        fn = shape_sdf(kind, P)
        surf = surface_samples(fn, lo, hi, n_near, rng)
        r = rng.random(n_near)
        sigma = np.select([r < 0.5, r < 0.8], [0.0005, 0.0015], 0.004)[:, None]
        near = surf + rng.normal(0, 1, surf.shape) * sigma
        far = rng.uniform(lo - 0.01, hi + 0.01, size=(points_per_shape - n_near, 3))
        pts = np.concatenate([near, far])
        pts_all.append(network_inputs(kind, P, pts))
        sdf_all.append((fn(pts) / SCALE).astype(np.float32))
        idx_all.append(np.full(len(pts), s, dtype=np.int32))
        conds.append(condition_vector(kind, P))
    return start, np.concatenate(pts_all), np.concatenate(sdf_all), np.concatenate(idx_all), np.stack(conds)


def make_dataset(n_shapes=6000, points_per_shape=10_000, seed=0, workers=None, verbose=True):
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    step = max(1, n_shapes // (workers * 6))
    tasks = [(a, min(a + step, n_shapes), points_per_shape, seed * 100_003 + a) for a in range(0, n_shapes, step)]
    parts, done, t0 = [], 0, time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        for part in ex.map(_build_chunk, tasks):
            parts.append(part)
            done += len(part[4])
            if verbose:
                rate = done / (time.perf_counter() - t0)
                print(f"  objects ready: {done}/{n_shapes}   ~{(n_shapes - done) / rate:.0f} s left", flush=True)
    parts.sort(key=lambda p: p[0])
    return dict(points=np.concatenate([p[1] for p in parts]), sdf=np.concatenate([p[2] for p in parts]),
                shape=np.concatenate([p[3] for p in parts]), cond=np.concatenate([p[4] for p in parts]),
                kinds=np.array([KINDS[s % len(KINDS)] for s in range(n_shapes)]))
