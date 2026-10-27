import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection

BACKGROUND = "black"
TEXT = "white"
POSITIVE = "#3987e5"
NEGATIVE = "#e66767"
FIXED = "#8a8a85"
OFF = "#444441"
PLAIN = "#1baf7a"
OUTLINE = "#b4b2a9"


def show_net_3d(model, x0=None, separate_window=True, max_lines=2000, save=None):
    if isinstance(model, nn.Linear):
        model = nn.Sequential(model)
    mods = list(model)
    linears = [m for m in mods if isinstance(m, nn.Linear)]
    sizes = [linears[0].in_features] + [lin.out_features for lin in linears]
    last = len(sizes) - 1
    gap = max(3.0, 0.6 * max(math.ceil(math.sqrt(n)) for n in sizes))

    layers = []
    for k, n in enumerate(sizes):
        name = "вход" if k == 0 else ("выход" if k == last else f"скрытый {k}")
        layers.append(dict(name=name, size=n, x=k * gap, z=0.0))
    links = [dict(src=k, dst=k + 1, W=lin.weight) for k, lin in enumerate(linears)]

    acts = None
    if x0 is not None:
        acts = [x0[0]]
        h = x0
        with torch.no_grad():
            for i, m in enumerate(mods):
                h = m(h)
                nxt = mods[i + 1] if i + 1 < len(mods) else None
                if nxt is None or isinstance(nxt, nn.Linear):
                    acts.append(h[0])

    title = f"вход: {[round(v, 2) for v in x0[0].tolist()]}" if x0 is not None else None
    show_graph_3d(layers, links, acts, title, separate_window, max_lines, save)


def show_graph_3d(layers, links, acts=None, title=None, separate_window=True, max_lines=2000, save=None,
                  view=(15, -70)):
    sizes = [layer["size"] for layer in layers]
    pos = []
    for layer in layers:
        n = layer["size"]
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        pos.append([(layer["x"], c - (cols - 1) / 2, layer["z"] + (rows - 1) / 2 - r)
                    for r, c in (divmod(i, cols) for i in range(n))])

    if separate_window and save is None:
        plt.switch_backend("TkAgg")
    fig = plt.figure(figsize=(12, 7), facecolor=BACKGROUND)
    ax = fig.add_subplot(projection="3d")
    ax.set_facecolor(BACKGROUND)

    for link in links:
        W = link["W"].detach().cpu()
        src, dst = pos[link["src"]], pos[link["dst"]]
        flat = W.abs().flatten()
        keep = torch.topk(flat, min(max_lines, int((flat > 0).sum()))).indices
        wmax = flat.max().item() + 1e-9
        segments, colors, widths = [], [], []
        for idx in keep.tolist():
            j, i = divmod(idx, W.shape[1])
            w = W[j, i].item()
            segments.append([src[i], dst[j]])
            if link.get("fixed"):
                colors.append(FIXED)
                widths.append(0.4)
            else:
                colors.append(POSITIVE if w > 0 else NEGATIVE)
                widths.append(0.3 + 2.5 * abs(w) / wmax)
        alpha = link.get("alpha", 0.35 if link.get("fixed") else 0.6)
        lines = Line3DCollection(segments, colors=colors, linewidths=widths, alpha=alpha)
        lines.set_clip_on(False)
        ax.add_collection3d(lines)

    cmap = matplotlib.colormaps["viridis"]
    has_in = {link["dst"] for link in links}
    has_out = {link["src"] for link in links}
    texts, text_xyz, biggest = [], [], 0.0
    for k, (layer, layer_pos) in enumerate(zip(layers, pos)):
        xs, ys, zs = zip(*layer_pos)
        n = len(xs)
        size = 220 if n <= 64 else max(8, 220 * 64 / n)
        biggest = max(biggest, size)
        if acts is None:
            colors = [to_rgba(PLAIN)] * n
        else:
            a = acts[k].detach().cpu()
            vmin, vmax = a.min().item(), a.max().item()
            span = vmax - vmin
            hidden = k in has_in and k in has_out
            colors = []
            for v in a.tolist():
                if hidden and v == 0:
                    colors.append(to_rgba(OFF))
                else:
                    norm = (v - vmin) / span if span > 1e-9 else 1.0
                    colors.append(cmap(0.15 + 0.85 * norm))
        balls = ax.scatter(xs, ys, zs, s=size, c=colors, edgecolors=OUTLINE,
                           linewidths=0.5 if n <= 64 else 0, depthshade=False)
        balls.set_clip_on(False)
        text_xyz.append((layer["x"], 0, max(zs) + 1.0))
        texts.append(ax.text(*text_xyz[-1], f"{layer['name']} ({sizes[k]})", ha="center",
                             color=TEXT, fontsize=10 if len(layers) <= 6 else 8))

    every = [p for layer_pos in pos for p in layer_pos]
    lo = [min(p[d] for p in every) - 1.0 for d in range(3)]
    hi = [max(p[d] for p in every) + 1.0 for d in range(3)]
    hi[2] += 1.5
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    aspect = [h - l for l, h in zip(lo, hi)]
    aspect[0] = min(aspect[0], 2.5 * max(aspect[1], aspect[2]))
    state = dict(zoom=1.0)
    ax.set_box_aspect(aspect, zoom=state["zoom"])
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95)
    ax.view_init(*view)
    ax.set_axis_off()
    if title:
        ax.set_title(title, color=TEXT)

    xyz = np.array(every).T
    text_xyz = np.array(text_xyz).T
    pad = math.sqrt(biggest) / 2 * fig.dpi / 72 + 8

    def on_screen(points):
        x2, y2, _ = proj3d.proj_transform(*points, ax.get_proj())
        return ax.transData.transform(np.column_stack([x2, y2]))

    def fit(event):
        px, tpx = on_screen(xyz), on_screen(text_xyz)
        tw = np.array([t.get_window_extent(event.renderer).width for t in texts])
        th = np.array([t.get_window_extent(event.renderer).height for t in texts])
        left = min(px[:, 0].min() - pad, (tpx[:, 0] - tw / 2).min())
        right = max(px[:, 0].max() + pad, (tpx[:, 0] + tw / 2).max())
        bottom = min(px[:, 1].min() - pad, tpx[:, 1].min())
        top = max(px[:, 1].max() + pad, (tpx[:, 1] + th).max())
        area = ax.get_position(original=True).transformed(fig.transFigure)
        cx, cy = (area.x0 + area.x1) / 2, (area.y0 + area.y1) / 2
        scale = min(area.width / 2 / max(cx - left, right - cx), area.height / 2 / max(cy - bottom, top - cy))
        if scale < 1 or scale > 1.06:
            state["zoom"] = min(max(state["zoom"] * scale * 0.98, 0.1), 10.0)
            ax.set_box_aspect(aspect, zoom=state["zoom"])
            if save is None:
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("draw_event", fit)
    if save is not None:
        for _ in range(6):
            fig.canvas.draw()
        fig.savefig(save, facecolor=BACKGROUND)
        plt.close(fig)
    else:
        plt.show()
