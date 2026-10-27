import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from skimage import measure

from .condition import CLAMP, COND_DIM, COND_SLOTS, N_INPUTS, SCALE, condition_vector, network_inputs
from .mesh import grid_voxel
from .shapes import KINDS, MESH_VOXEL, shape_bbox

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "kitchgen_sdf.pt"
CHECKPOINT_PATH = ROOT / "models" / "kitchgen_sdf_training.pt"
DATA_PATH = ROOT / "data" / "kitchgen_sdf_dataset.npz"


class FourierFeatures(nn.Module):
    def __init__(self, n_freqs=10):
        super().__init__()
        self.register_buffer("freqs", np.pi * 2.0 ** torch.arange(n_freqs))

    def forward(self, p):
        x = p[..., None] * self.freqs
        return torch.cat([p, torch.sin(x).flatten(1), torch.cos(x).flatten(1)], dim=-1)


class ConditionalSDF(nn.Module):
    def __init__(self, width=256, depth=6, n_freqs=10, cond_width=64):
        super().__init__()
        self.ff = FourierFeatures(n_freqs)
        self.cond_net = nn.Sequential(nn.Linear(COND_DIM, 128), nn.SiLU(), nn.Linear(128, cond_width))
        in_dim = N_INPUTS + 2 * N_INPUTS * n_freqs + cond_width
        self.skip_at = depth // 2
        self.layers = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                d_in = in_dim
            elif i == self.skip_at:
                d_in = width + in_dim
            else:
                d_in = width + cond_width
            self.layers.append(nn.Linear(d_in, width))
        self.out = nn.Linear(width, 1)
        self.act = nn.SiLU()

    def forward(self, p, cond):
        c = self.cond_net(cond)
        if c.shape[0] == 1:
            c = c.expand(len(p), -1)
        x0 = torch.cat([self.ff(p), c], dim=-1)
        h = x0
        for i, layer in enumerate(self.layers):
            if i == 0:
                inp = x0
            elif i == self.skip_at:
                inp = torch.cat([h, x0], dim=-1)
            else:
                inp = torch.cat([h, c], dim=-1)
            h = self.act(layer(inp))
        return self.out(h).squeeze(-1)


def train(data, steps=80_000, batch=32_768, lr=5e-4, val_shapes=36, device=None, log_every=2000,
          checkpoint=None, save_every=2000):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    n_shapes = len(data["cond"])
    val_ids = np.arange(n_shapes - val_shapes, n_shapes)
    is_val = np.isin(data["shape"], val_ids)

    pts = torch.from_numpy(data["points"]).to(device)
    sdf = torch.from_numpy(data["sdf"]).to(device).clamp(-CLAMP, CLAMP)
    shape = torch.from_numpy(data["shape"]).long().to(device)
    cond = torch.from_numpy(data["cond"]).to(device)
    train_idx = torch.from_numpy(np.flatnonzero(~is_val)).to(device)
    val_idx = torch.from_numpy(np.flatnonzero(is_val)).to(device)

    model = ConditionalSDF().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    start = 1
    if checkpoint and Path(checkpoint).exists():
        ckpt = torch.load(checkpoint, map_location=device)
        if ckpt["steps"] == steps and ckpt["n_shapes"] == n_shapes:
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            sched.load_state_dict(ckpt["sched"])
            start = ckpt["step"] + 1
            print(f"Resuming the interrupted training from step {ckpt['step']}", flush=True)
    for step in range(start, steps + 1):
        i = train_idx[torch.randint(len(train_idx), (batch,), device=device)]
        loss = (model(pts[i], cond[shape[i]]) - sdf[i]).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if checkpoint and step % save_every == 0:
            tmp = Path(checkpoint).with_suffix(".tmp")
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step,
                            steps=steps, n_shapes=n_shapes), tmp)
            os.replace(tmp, checkpoint)
        if step % log_every == 0 or step == start:
            with torch.no_grad():
                j = val_idx[torch.randint(len(val_idx), (batch,), device=device)]
                vloss = (model(pts[j], cond[shape[j]]) - sdf[j]).abs().mean()
            print(f"step {step:6d}/{steps}   error on training objects {loss.item() * SCALE * 1000:.3f} mm"
                  f"   on unseen objects {vloss.item() * SCALE * 1000:.3f} mm", flush=True)
    return model


def save_model(model, path=MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=model.state_dict(), cond_slots=COND_SLOTS, kinds=KINDS, scale=SCALE, inputs=N_INPUTS), path)


def load_model(path=MODEL_PATH, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not Path(path).exists():
        raise FileNotFoundError(f"No trained model at {path}. Train one first: python train.py")
    ckpt = torch.load(path, map_location=device)
    if ckpt.get("cond_slots") != COND_SLOTS or ckpt.get("inputs", 3) != N_INPUTS:
        raise RuntimeError("The saved model was trained for a different set of objects. Retrain: python train.py --rebuild")
    model = ConditionalSDF().to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model


@torch.no_grad()
def neural_mesh(model, kind, params, chunk=262_144):
    device = next(model.parameters()).device
    lo, hi = shape_bbox(kind, params)
    lo, hi = lo - 0.004, hi + 0.004
    voxel = grid_voxel(lo, hi, MESH_VOXEL[kind])
    axes = [np.arange(lo[i], hi[i] + voxel, voxel) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], -1)
    c = torch.from_numpy(condition_vector(kind, params)).to(device)[None]
    values = np.empty(len(grid), dtype=np.float32)
    for i in range(0, len(grid), chunk):
        x = torch.from_numpy(network_inputs(kind, params, grid[i:i + chunk])).to(device)
        values[i:i + chunk] = model(x, c).cpu().numpy()
    values = values.reshape(gx.shape)
    if values.min() > 0:
        raise RuntimeError("The network drew no surface: is the model trained?")
    verts, faces, _, _ = measure.marching_cubes(values, level=0.0, spacing=(voxel,) * 3)
    return _largest_parts(verts + lo, faces.astype(np.int64))


def _largest_parts(verts, faces, keep_ratio=0.02):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(verts)
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    _, labels = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)), directed=False)
    sizes = np.bincount(labels)
    keep_v = sizes[labels] >= keep_ratio * sizes.max()
    remap = -np.ones(n, dtype=np.int64)
    remap[keep_v] = np.arange(keep_v.sum())
    return verts[keep_v], remap[faces[keep_v[faces[:, 0]]]]
