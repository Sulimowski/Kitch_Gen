import argparse

import numpy as np
import torch

from generate import make_params, request_fixed
from kitchgen.condition import N_INPUTS, SCALE, condition_vector, network_inputs
from kitchgen.dataset import surface_samples
from kitchgen.neural import ConditionalSDF, load_model
from kitchgen.shapes import DISPLAY_NAMES, shape_bbox, shape_sdf
from kitchgen.text import describe_kinds, parse
from netviz import show_graph_3d


def fourier_wiring(model):
    n_in, n_freqs = N_INPUTS, len(model.ff.freqs)
    src = list(range(n_in)) + [i for i in range(n_in) for _ in range(n_freqs)] * 2
    W = torch.zeros(len(src), n_in)
    W[torch.arange(len(src)), torch.tensor(src)] = 1.0
    return W


def network_graph(model: ConditionalSDF):
    gap, below = 22.0, -30.0
    width = model.out.in_features
    n_ff = model.layers[0].in_features - model.cond_net[-1].out_features
    n_code = model.cond_net[-1].out_features
    layers = [dict(name="point (x, y, z, warp)", size=N_INPUTS, x=0.0, z=0.0),
              dict(name="Fourier features", size=n_ff, x=gap, z=0.0)]
    layers += [dict(name=f"hidden {i + 1}", size=width, x=(2 + i) * gap, z=0.0) for i in range(len(model.layers))]
    out = len(layers)
    layers += [dict(name="distance", size=1, x=(2 + len(model.layers)) * gap, z=0.0),
               dict(name="description", size=model.cond_net[0].in_features, x=0.0, z=below),
               dict(name="description layer", size=model.cond_net[0].out_features, x=gap, z=below),
               dict(name="description code", size=n_code, x=2 * gap, z=below)]
    desc, desc_hidden, code = out + 1, out + 2, out + 3

    links = [dict(src=0, dst=1, W=fourier_wiring(model), fixed=True),
             dict(src=desc, dst=desc_hidden, W=model.cond_net[0].weight),
             dict(src=desc_hidden, dst=code, W=model.cond_net[2].weight)]
    for i, layer in enumerate(model.layers):
        W, h = layer.weight, 2 + i
        if i == 0:
            links += [dict(src=1, dst=h, W=W[:, :n_ff]), dict(src=code, dst=h, W=W[:, n_ff:], alpha=0.2)]
        elif i == model.skip_at:
            links += [dict(src=h - 1, dst=h, W=W[:, :width]), dict(src=1, dst=h, W=W[:, width:width + n_ff]),
                      dict(src=code, dst=h, W=W[:, width + n_ff:], alpha=0.2)]
        else:
            links += [dict(src=h - 1, dst=h, W=W[:, :width]), dict(src=code, dst=h, W=W[:, width:], alpha=0.2)]
    links.append(dict(src=out - 1, dst=out, W=model.out.weight))
    return layers, links


@torch.no_grad()
def activations(model, x, cond):
    feats = model.ff(x)
    desc_hidden = model.cond_net[1](model.cond_net[0](cond))
    code = model.cond_net[2](desc_hidden)
    x0 = torch.cat([feats, code], -1)
    h, hidden = x0, []
    for i, layer in enumerate(model.layers):
        inp = x0 if i == 0 else torch.cat([h, x0] if i == model.skip_at else [h, code], -1)
        h = model.act(layer(inp))
        hidden.append(h[0])
    dist = model.out(h)
    return [x[0], feats[0], *hidden, dist[0], cond[0], desc_hidden[0], code[0]], dist.item()


def main():
    ap = argparse.ArgumentParser(description="3D snapshot of the KitchGen neural network")
    ap.add_argument("text", nargs="?", default="square mug", help="the object the network is looking at")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lines", type=int, default=150, help="strongest links drawn between each pair of layers")
    ap.add_argument("--save", default=None, help="save a picture instead of opening a window")
    args = ap.parse_args()

    req = parse(args.text)
    if req.kind is None:
        print(f"Could not tell which object you mean. Supported: {describe_kinds()}.")
        return
    if req.cross_kind:
        print("The network has only learned single objects; showing it the first one.")
        req.cross_kind = None
    rng = np.random.default_rng(args.seed)
    params = make_params(req.kind, req, request_fixed(req, rng), rng)
    lo, hi = shape_bbox(req.kind, params)
    point = surface_samples(shape_sdf(req.kind, params), lo, hi, 1, rng)

    model = load_model(device="cpu")
    x = torch.from_numpy(network_inputs(req.kind, params, point))
    cond = torch.from_numpy(condition_vector(req.kind, params))[None]
    acts, dist = activations(model, x, cond)
    true = shape_sdf(req.kind, params)(point)[0]
    print(f"Point on a {DISPLAY_NAMES[req.kind]}: {np.round(point[0] * 100, 1)} cm, "
          f"network says {dist * SCALE * 1000:+.2f} mm from the surface (truth {true * 1000:+.2f} mm)")

    layers, links = network_graph(model)
    title = (f"'{args.text}': a point on the surface -> {dist * SCALE * 1000:+.2f} mm "
             f"(truth {true * 1000:+.2f} mm)")
    show_graph_3d(layers, links, acts, title, max_lines=args.lines, save=args.save, view=(14, -40))
    if args.save:
        print(f"Saved: {args.save}")


if __name__ == "__main__":
    main()
