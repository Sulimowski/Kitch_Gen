import argparse
import time
from pathlib import Path

import numpy as np

from kitchgen.dataset import make_dataset

DATA_PATH = Path(__file__).resolve().parent / "data" / "kitchgen_sdf_dataset.npz"


def compare_picture(model, path, seed=123):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from kitchgen.mesh import sdf_to_mesh
    from kitchgen.neural import neural_mesh
    from kitchgen.shapes import DISPLAY_NAMES, KINDS, MESH_VOXEL, random_params, shape_bbox, shape_sdf
    rng = np.random.default_rng(seed)
    cols = 4
    rows = (len(KINDS) + 1) // 2
    fig = plt.figure(figsize=(16, 3.4 * rows), facecolor="black")
    for k, kind in enumerate(KINDS):
        P = random_params(kind, rng)
        lo, hi = shape_bbox(kind, P)
        meshes = [sdf_to_mesh(shape_sdf(kind, P), lo, hi, voxel=MESH_VOXEL[kind]), neural_mesh(model, kind, P)]
        for j, (v, f) in enumerate(meshes):
            ax = fig.add_subplot(rows, cols, 2 * k + j + 1, projection="3d")
            ax.set_facecolor("black")
            ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f, color="#c9cad3", linewidth=0)
            ax.set_box_aspect(np.ptp(v, axis=0))
            ax.view_init(elev=30, azim=-60)
            ax.set_axis_off()
            ax.set_title(f"{DISPLAY_NAMES[kind]}: {'procedural' if j == 0 else 'neural'}", color="white")
    fig.tight_layout()
    fig.savefig(path, dpi=70, facecolor="black")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", type=int, default=6000, help="objects in the training data")
    ap.add_argument("--points", type=int, default=10_000, help="points per object")
    ap.add_argument("--steps", type=int, default=80_000)
    ap.add_argument("--rebuild", action="store_true", help="rebuild the training data")
    args = ap.parse_args()

    from kitchgen.condition import N_INPUTS
    from kitchgen.neural import CHECKPOINT_PATH
    data = dict(np.load(DATA_PATH)) if DATA_PATH.exists() and not args.rebuild else None
    if data is not None and data["points"].shape[1] != N_INPUTS:
        print("The saved training data is from an older version of the network, rebuilding it.", flush=True)
        data = None
    if data is not None:
        print(f"Using existing training data: {DATA_PATH}", flush=True)
    else:
        CHECKPOINT_PATH.unlink(missing_ok=True)
        print(f"Building training data: {args.shapes} objects x {args.points} points...", flush=True)
        t = time.perf_counter()
        data = make_dataset(args.shapes, args.points)
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(DATA_PATH, **data)
        print(f"Training data ready in {time.perf_counter() - t:.0f} s", flush=True)

    from kitchgen.neural import MODEL_PATH, save_model, train
    t = time.perf_counter()
    model = train(data, steps=args.steps, checkpoint=CHECKPOINT_PATH)
    print(f"Training took {time.perf_counter() - t:.0f} s", flush=True)
    save_model(model)
    CHECKPOINT_PATH.unlink(missing_ok=True)
    print(f"Model saved: {MODEL_PATH}", flush=True)

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    compare_picture(model, out / "neural_check.png")
    print(f"Shape comparison: {(out / 'neural_check.png').resolve()}", flush=True)


if __name__ == "__main__":
    main()
