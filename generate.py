import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from kitchgen.hybrid import can_cross, cross
from kitchgen.mesh import save_obj, save_ply, sdf_to_mesh
from kitchgen.shapes import (DISPLAY_NAMES, MATERIAL, MESH_VOXEL, MOD_NEUTRAL, random_params, scale_to_length,
                             set_capacity, set_diameter, set_height, shape_bbox, shape_sdf)
from kitchgen.stereo import capture
from kitchgen.text import describe_kinds, parse
from kitchgen.viz import save_depth_png, view_html

RATIO_PARAMS = {"taper", "belly", "roundness", "rim_ratio", "flare", "handle_size", "handle_height", "rim_width",
                "tip_round", "squareness", "aspect", "lobe_depth", "bulge"}
ANGLE_PARAMS = {"handle_angle", "twist"}


def describe_value(kind, name, value):
    if isinstance(value, int):
        return str(value)
    if name in ANGLE_PARAMS:
        return f"{np.degrees(value):.0f} deg"
    if name in RATIO_PARAMS or (kind in ("bowl", "plate") and name == "foot_radius") \
            or (kind == "ladle" and name == "bowl_depth"):
        return f"{value:.2f}"
    return f"{value * 100:.1f} cm"


def make_params(kind, req, fixed, rng):
    params = random_params(kind, rng, fixed)
    if req.length:
        scale_to_length(kind, params, req.length)
    if req.height:
        set_height(kind, params, req.height)
    if req.diameter:
        set_diameter(kind, params, req.diameter)
    if req.capacity:
        set_capacity(kind, params, req.capacity)
    return params


def request_fixed(req, rng, wild=False):
    fixed = req.fixed_params(rng)
    if not wild and not any(name in fixed for name in MOD_NEUTRAL):
        fixed = {**MOD_NEUTRAL, **fixed}
    return fixed


def build_shape(req, rng, model=None, wild=False):
    fixed = request_fixed(req, rng, wild)
    params = make_params(req.kind, req, fixed, rng)
    if req.cross_kind:
        params_b = make_params(req.cross_kind, req, fixed, rng)
        fn, lo, hi = cross(req.kind, params, req.cross_kind, params_b, req.cross_weight)
        verts, faces = sdf_to_mesh(fn, lo, hi, voxel=min(MESH_VOXEL[req.kind], MESH_VOXEL[req.cross_kind]))
        return params, params_b, verts, faces
    if model is not None:
        from kitchgen.neural import neural_mesh
        verts, faces = neural_mesh(model, req.kind, params)
    else:
        lo, hi = shape_bbox(req.kind, params)
        verts, faces = sdf_to_mesh(shape_sdf(req.kind, params), lo, hi, voxel=MESH_VOXEL[req.kind])
    return params, None, verts, faces


def print_params(kind, params):
    for name, value in asdict(params).items():
        if name in MOD_NEUTRAL and value == MOD_NEUTRAL[name]:
            continue
        print(f"  {name:17s} {describe_value(kind, name, value)}")


def save_sample(out, args, req, params, params_b, verts, faces, captures, source):
    out.mkdir(parents=True, exist_ok=True)
    save_obj(out / "mesh.obj", verts, faces)
    if len(captures) == 1:
        save_ply(out / "cloud.ply", captures[0]["points"])
        save_ply(out / "cloud_camera.ply", captures[0]["points_cam"])
        save_depth_png(out / "depth.png", captures[0]["depth"])
    else:
        save_ply(out / "cloud.ply", np.concatenate([c["points"] for c in captures]))
        for k, cap in enumerate(captures):
            save_ply(out / f"cloud_view{k + 1}.ply", cap["points"])
            save_ply(out / f"cloud_camera_view{k + 1}.ply", cap["points_cam"])
            save_depth_png(out / f"depth_view{k + 1}.png", cap["depth"])
    cameras = [dict(asdict(c["camera"]), eye=c["eye"].tolist(), rotation_world_to_camera=c["R"].tolist(),
                    points=len(c["points"]), points_on_object=int(c["on_object"].sum())) for c in captures]
    info = dict(text=args.text, kind=req.kind, source=source, seed=args.seed, material=captures[0]["material"],
                understood=req.notes, params=asdict(params), cameras=cameras)
    if params_b is not None:
        info.update(cross_kind=req.cross_kind, cross_weight=req.cross_weight, cross_params=asdict(params_b))
    (out / "params.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Kitchenware generator: text → mesh + stereo point cloud")
    ap.add_argument("text", help='description, e.g. "fork with 3 prongs and a long handle"')
    ap.add_argument("--seed", type=int, default=None, help="random seed: the same seed gives the same result")
    ap.add_argument("--table", action="store_true", help="put the object on a table (the table is in the cloud too)")
    ap.add_argument("--views", type=int, default=1, help="how many cameras shoot the object")
    ap.add_argument("--count", type=int, default=1, help="how many different objects to generate")
    ap.add_argument("--neural", action="store_true", help="let the trained neural network build the shape")
    ap.add_argument("--wild", action="store_true",
                    help="containers also get a random outline (square, oval, wavy, twisted...) unless the text sets one")
    ap.add_argument("--no-view", action="store_true", help="do not open the browser")
    ap.add_argument("--out", default="outputs", help="output folder")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    req = parse(args.text)
    if req.kind is None:
        print(f"Could not tell which object you mean. Supported: {describe_kinds()}.")
        return
    if req.cross_kind and not can_cross(req.kind, req.cross_kind):
        print(f"A {DISPLAY_NAMES[req.kind]} and a {DISPLAY_NAMES[req.cross_kind]} cannot be crossed: crosses work "
              f"between containers (mug, cup, glass, wine glass, bowl, plate, frying pan) or between fork, spoon, "
              f"knife and spatula. Making a plain {DISPLAY_NAMES[req.kind]}.")
        req.cross_kind = None
        req.notes = [n for n in req.notes if not n.startswith("cross with")]
    material = req.material or MATERIAL[req.kind]
    name = DISPLAY_NAMES[req.kind] + (f"-{DISPLAY_NAMES[req.cross_kind]} cross" if req.cross_kind else "")
    print(f"Object: {name} ({material})")
    print("Understood:", ", ".join(req.notes) if req.notes else "nothing specific, the rest is random")

    model = None
    if args.neural and req.cross_kind:
        print("The neural network has only learned single objects, so the cross is built procedurally.")
    elif args.neural:
        from kitchgen.neural import load_model
        model = load_model()
    source = "neural network" if model is not None else "procedural generator"

    tag = req.kind + (f"-{req.cross_kind}" if req.cross_kind else "")
    run = Path(args.out) / f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    for i in range(args.count):
        t0 = time.perf_counter()
        params, params_b, verts, faces = build_shape(req, rng, model, wild=args.wild)
        if i == 0:
            print_params(req.kind, params)
            if params_b is not None:
                print(f"crossed with a {DISPLAY_NAMES[req.cross_kind]} ({100 * req.cross_weight:.0f}%):")
                print_params(req.cross_kind, params_b)
            lo, hi = verts.min(0), verts.max(0)
            print(f"  {'size':17s} {(hi[0] - lo[0]) * 100:.1f} x {(hi[1] - lo[1]) * 100:.1f} x "
                  f"{(hi[2] - lo[2]) * 100:.1f} cm")
        captures = [capture(verts, faces, rng, material=material, table=args.table) for _ in range(args.views)]
        out = run if args.count == 1 else run / f"sample_{i:03d}"
        save_sample(out, args, req, params, params_b, verts, faces, captures, source)
        n_obj = sum(int(c["on_object"].sum()) for c in captures)
        print(f"[{i + 1}/{args.count}] mesh ({source}): {len(faces)} triangles, "
              f"stereo: {sum(len(c['points']) for c in captures)} points ({n_obj} on the object), "
              f"{time.perf_counter() - t0:.1f} s")
        if i == 0:
            title = f"{name}: '{args.text}' ({source})"
            view_html(out / "view.html", verts, faces, captures, title, open_browser=not args.no_view, rng=rng)

    print(f"Done: {run.resolve()}")
    print("  mesh.obj - mesh, cloud.ply - stereo point cloud, depth.png - depth frame, view.html - 3D viewer")


if __name__ == "__main__":
    main()
