import numpy as np
from skimage import measure

VOXEL = 0.0007
MAX_GRID_POINTS = 12e6


def grid_voxel(lo, hi, voxel=VOXEL):
    volume = float(np.prod(np.asarray(hi) - np.asarray(lo)))
    return max(voxel, (volume / MAX_GRID_POINTS) ** (1 / 3))


def sdf_grid(sdf_fn, lo, hi, voxel, chunk=400_000):
    axes = [np.arange(lo[i], hi[i] + voxel, voxel) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
    values = np.empty(len(pts))
    for i in range(0, len(pts), chunk):
        values[i:i + chunk] = sdf_fn(pts[i:i + chunk])
    return values.reshape(gx.shape)


def sdf_to_mesh(sdf_fn, lo, hi, voxel=None):
    voxel = voxel or grid_voxel(lo, hi)
    values = sdf_grid(sdf_fn, lo, hi, voxel)
    verts, faces, _, _ = measure.marching_cubes(values, level=0.0, spacing=(voxel,) * 3)
    return verts + np.asarray(lo), faces.astype(np.int64)


def save_obj(path, verts, faces):
    with open(path, "w", encoding="utf-8") as f:
        np.savetxt(f, verts, fmt="v %.6f %.6f %.6f")
        np.savetxt(f, faces + 1, fmt="f %d %d %d")


def save_ply(path, points):
    header = ("ply\nformat ascii 1.0\n"
              f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\nend_header")
    np.savetxt(path, points, fmt="%.6f", header=header, comments="")
