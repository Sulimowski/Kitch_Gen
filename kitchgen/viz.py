import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

MAX_SHOWN_POINTS = 150_000
VIEW_COLORS = ["#f2a623", "#3987e5", "#1baf7a", "#e66767", "#9085e9", "#d55181"]


def _mesh_trace(verts, faces):
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color="#c9cad3", name="mesh", showlegend=True,
        lighting=dict(ambient=0.35, diffuse=0.8, specular=0.9, roughness=0.25, fresnel=0.3),
        lightposition=dict(x=100, y=200, z=300))


def view_html(path, verts=None, faces=None, captures=(), title="", open_browser=True, rng=None):
    rng = rng or np.random.default_rng(0)
    traces = [_mesh_trace(verts, faces)] if verts is not None else []
    per_view = MAX_SHOWN_POINTS // max(len(captures), 1)
    for k, cap in enumerate(captures):
        pts = cap["points"]
        if len(pts) > per_view:
            pts = pts[rng.choice(len(pts), per_view, replace=False)]
        dist = np.linalg.norm(pts - cap["eye"], axis=1)
        color = dict(color=dist, colorscale="Turbo") if len(captures) == 1 else dict(color=VIEW_COLORS[k % 6])
        name = "stereo point cloud" if len(captures) == 1 else f"stereo view {k + 1}"
        traces.append(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", name=name,
                                   marker=dict(size=1.6, showscale=False, **color)))
        eye, target = cap["eye"], cap["target"]
        traces.append(go.Scatter3d(
            x=[eye[0], target[0]], y=[eye[1], target[1]], z=[eye[2], target[2]], mode="lines+markers",
            name=f"camera {k + 1}" if len(captures) > 1 else "camera",
            line=dict(color=VIEW_COLORS[k % 6], width=3), marker=dict(size=[6, 0], color=VIEW_COLORS[k % 6])))

    fig = go.Figure(traces)
    axis = dict(visible=False)
    fig.update_layout(
        title=dict(text=title, font=dict(color="white")),
        paper_bgcolor="black", plot_bgcolor="black",
        legend=dict(font=dict(color="white"), itemsizing="constant"),
        scene=dict(bgcolor="black", aspectmode="data", xaxis=axis, yaxis=axis, zaxis=axis),
        margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(str(path))
    if open_browser:
        webbrowser.open(Path(path).resolve().as_uri())
    return path


def save_depth_png(path, depth):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(depth.shape[1] / 100, depth.shape[0] / 100), facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(depth, cmap="turbo")
    ax.set_facecolor("black")
    ax.axis("off")
    fig.savefig(path, dpi=100, facecolor="black")
