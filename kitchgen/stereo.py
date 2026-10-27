from dataclasses import dataclass

import numpy as np

MATERIALS = {
    "metal":   dict(specular_cos=0.97, specular_drop=0.6, dropout=0.03, see_through=0.0),
    "ceramic": dict(specular_cos=0.995, specular_drop=0.35, dropout=0.012, see_through=0.0),
    "plastic": dict(specular_cos=0.99, specular_drop=0.25, dropout=0.015, see_through=0.0),
    "glass":   dict(specular_cos=0.9, specular_drop=0.75, dropout=0.35, see_through=0.55),
}
TABLE_DROPOUT = 0.02


@dataclass
class StereoCamera:
    width: int = 1280
    height: int = 720
    fx: float = 915.0
    fy: float = 915.0
    baseline: float = 0.05
    disp_noise: float = 0.25
    subpixel: float = 1 / 8
    min_depth: float = 0.15
    max_depth: float = 2.0
    edge_jump: float = 0.01
    flying_prob: float = 0.5

    @property
    def cx(self):
        return self.width / 2

    @property
    def cy(self):
        return self.height / 2


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    return np.stack([r, d, f])


def random_eye(rng, target, radius, cam, elevation_deg=(30, 75)):
    half_fov = np.arctan(cam.height / 2 / cam.fy)
    fit = radius / np.sin(half_fov)
    dist = float(np.clip(fit * rng.uniform(0.85, 1.15), 0.28, 1.2))
    az = rng.uniform(0, 2 * np.pi)
    el = np.radians(rng.uniform(*elevation_deg))
    return np.asarray(target) + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def triangle_areas(verts, faces):
    tri = verts[faces]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def sample_surface(verts, faces, n, rng, areas=None):
    areas = triangle_areas(verts, faces) if areas is None else areas
    tri = verts[faces]
    idx = rng.choice(len(faces), size=n, p=areas / areas.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[idx]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


def zbuffer(pc, cam):
    z = pc[:, 2]
    ok = z > 1e-4
    u = np.round(cam.fx * pc[ok, 0] / z[ok] + cam.cx).astype(int)
    v = np.round(cam.fy * pc[ok, 1] / z[ok] + cam.cy).astype(int)
    inside = (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    depth = np.full(cam.height * cam.width, np.inf)
    np.minimum.at(depth, v[inside] * cam.width + u[inside], z[ok][inside])
    return depth.reshape(cam.height, cam.width)


def plane_depth(cam, R, eye, plane_z, center, half_size):
    u, v = np.meshgrid(np.arange(cam.width), np.arange(cam.height))
    dirs_cam = np.stack([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, np.ones_like(u, float)], axis=-1)
    dirs = dirs_cam @ R
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (plane_z - eye[2]) / dirs[..., 2]
    hit = eye + s[..., None] * dirs
    ok = (s > 0) & (np.abs(hit[..., 0] - center[0]) < half_size) & (np.abs(hit[..., 1] - center[1]) < half_size)
    return np.where(ok, s, np.inf)


def _neighbors(a, fill):
    p = np.pad(a, 1, constant_values=fill)
    return [p[1:-1, :-2], p[1:-1, 2:], p[:-2, 1:-1], p[2:, 1:-1]]


def capture(verts, faces, rng, cam=None, material="metal", table=False, eye=None):
    cam = cam or StereoCamera()
    mat = MATERIALS[material]
    lo, hi = verts.min(0), verts.max(0)
    target = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo) / 2)
    if eye is None:
        eye = random_eye(rng, target, radius, cam)
    eye = np.asarray(eye, float)
    R = look_at(eye, target)
    B = cam.baseline

    areas = triangle_areas(verts, faces)
    pixel_area = (np.linalg.norm(eye - target) / cam.fx) ** 2
    n_samples = int(np.clip(1.5 * areas.sum() / pixel_area, 300_000, 3_000_000))
    pts = sample_surface(verts, faces, n_samples, rng, areas)
    pc_left = (pts - eye) @ R.T
    pc_right = pc_left - np.array([B, 0.0, 0.0])
    obj_left, obj_right = zbuffer(pc_left, cam), zbuffer(pc_right, cam)

    depth_l, depth_r = obj_left, obj_right
    table_l = np.full_like(obj_left, np.inf)
    if table:
        plane_z, size = lo[2], max(0.35, 1.5 * radius)
        table_l = plane_depth(cam, R, eye, plane_z, target, size)
        depth_l = np.minimum(obj_left, table_l)
        depth_r = np.minimum(obj_right, plane_depth(cam, R, eye + B * R[0], plane_z, target, size))
    is_object = np.isfinite(obj_left) & (obj_left <= depth_l + 1e-9)

    H, W = depth_l.shape
    vv, uu = np.mgrid[0:H, 0:W]
    Z = depth_l.copy()
    valid = np.isfinite(Z) & (Z > cam.min_depth) & (Z < cam.max_depth)

    with np.errstate(divide="ignore", invalid="ignore"):
        ur = np.round(uu - cam.fx * B / Z).astype(int)
    in_right = valid & (ur >= 0) & (ur < W)
    seen = np.zeros_like(valid)
    tol = 0.003 + 0.01 * np.where(valid, Z, 0)
    seen[in_right] = depth_r[vv[in_right], ur[in_right]] >= Z[in_right] - tol[in_right]
    valid &= seen

    Zf = np.where(np.isfinite(depth_l), depth_l, np.nan)
    P = np.stack([(uu - cam.cx) * Zf / cam.fx, (vv - cam.cy) * Zf / cam.fy, Zf], axis=-1)
    du = np.zeros_like(P)
    dv = np.zeros_like(P)
    du[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dv[1:-1] = P[2:] - P[:-2]
    n = np.cross(du, dv)
    with np.errstate(invalid="ignore"):
        cosang = np.abs(np.sum(n * P, axis=-1)) / (np.linalg.norm(n, axis=-1) * np.linalg.norm(P, axis=-1) + 1e-12)
    specular = is_object & (np.nan_to_num(cosang) > mat["specular_cos"])
    valid &= ~(specular & (rng.random((H, W)) < mat["specular_drop"]))

    far = np.full_like(Z, -np.inf)
    empty_nbr = np.zeros_like(valid)
    for nb in _neighbors(depth_l, np.inf):
        far = np.where(np.isfinite(nb), np.maximum(far, nb), far)
        empty_nbr |= ~np.isfinite(nb)
    edge = valid & ((far - Z > cam.edge_jump) | empty_nbr)
    fly = edge & (rng.random((H, W)) < cam.flying_prob)
    can_mix = fly & np.isfinite(far) & (far - Z > cam.edge_jump)
    with np.errstate(invalid="ignore"):
        Z = np.where(can_mix, Z + rng.random((H, W)) * (far - Z), Z)
    valid &= ~(fly & ~can_mix)

    through = is_object & valid & (rng.random((H, W)) < mat["see_through"])
    Z = np.where(through, table_l, Z)
    valid &= np.isfinite(Z)
    is_object = is_object & ~through

    hole_rate = np.where(is_object, mat["dropout"], TABLE_DROPOUT)
    valid &= rng.random((H, W)) >= hole_rate

    disp = cam.fx * B / Z[valid]
    disp = disp + rng.normal(0.0, cam.disp_noise, disp.shape)
    disp = np.round(disp / cam.subpixel) * cam.subpixel
    keep = disp > 0
    z_noisy = cam.fx * B / disp[keep]
    u_px, v_px = uu[valid][keep], vv[valid][keep]
    points_cam = np.stack([(u_px - cam.cx) * z_noisy / cam.fx, (v_px - cam.cy) * z_noisy / cam.fy, z_noisy], -1)
    points_world = points_cam @ R + eye

    depth_out = np.full((H, W), np.nan)
    depth_out[v_px, u_px] = z_noisy
    return dict(points=points_world, points_cam=points_cam, on_object=is_object[v_px, u_px], depth=depth_out,
                eye=eye, target=target, R=R, camera=cam, material=material)
