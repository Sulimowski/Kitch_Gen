from dataclasses import asdict, dataclass, fields

import numpy as np

PAD = 0.003


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def smooth_min(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def tapered_capsule_2d(x, y, x0, x1, r0, r1):
    t = np.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    px = x - (x0 + t * (x1 - x0))
    return np.sqrt(px * px + y * y) - (r0 + t * (r1 - r0))


def extrude(d2, z, half, round_r=0.0):
    wx = d2 + round_r
    wz = np.abs(z) - half + round_r
    inside = np.minimum(np.maximum(wx, wz), 0.0)
    outside = np.sqrt(np.maximum(wx, 0.0) ** 2 + np.maximum(wz, 0.0) ** 2)
    return inside + outside - round_r


def ellipsoid(q, r):
    k0 = np.linalg.norm(q / r, axis=-1)
    k1 = np.linalg.norm(q / (r * r), axis=-1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-12)


def seg2(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    h = np.clip(((px - ax) * dx + (py - ay) * dy) / max(dx * dx + dy * dy, 1e-12), 0.0, 1.0)
    return np.hypot(px - ax - h * dx, py - ay - h * dy)


def polyline2(px, py, pts):
    d = np.full_like(px, np.inf)
    for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
        d = np.minimum(d, seg2(px, py, ax, ay, bx, by))
    return d


def polyline3(p, pts):
    d = np.full(len(p), np.inf)
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        h = np.clip((p - a) @ ab / max(ab @ ab, 1e-12), 0.0, 1.0)
        d = np.minimum(d, np.linalg.norm(p - a - h[:, None] * ab, axis=1))
    return d


def rounded_box(qx, qy, qz, hx, hy, hz, r):
    dx, dy, dz = np.abs(qx) - hx + r, np.abs(qy) - hy + r, np.abs(qz) - hz + r
    outside = np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2 + np.maximum(dz, 0) ** 2)
    return outside + np.minimum(np.maximum(dx, np.maximum(dy, dz)), 0.0) - r


def _handle_bend(x, start, handle_len, amount):
    return amount * (np.maximum(-x - start, 0.0) / handle_len) ** 2


@dataclass
class ForkParams:
    n_tines: int = 4
    handle_len: float = 0.110
    handle_width: float = 0.015
    neck_len: float = 0.020
    neck_width: float = 0.006
    head_width: float = 0.026
    base_len: float = 0.014
    tine_len: float = 0.032
    thickness: float = 0.003
    bend: float = 0.012


@dataclass
class SpoonParams:
    handle_len: float = 0.105
    handle_width: float = 0.016
    neck_len: float = 0.018
    neck_width: float = 0.006
    bowl_len: float = 0.050
    bowl_width: float = 0.034
    bowl_depth: float = 0.009
    thickness: float = 0.0025
    bend: float = 0.010


@dataclass
class KnifeParams:
    handle_len: float = 0.095
    handle_width: float = 0.017
    handle_thickness: float = 0.009
    blade_len: float = 0.115
    blade_width: float = 0.022
    blade_thickness: float = 0.0022
    tip_round: float = 0.3
    serrated: int = 0


@dataclass
class LadleParams:
    bowl_radius: float = 0.042
    bowl_depth: float = 0.8
    handle_len: float = 0.24
    handle_angle: float = 0.45
    handle_radius: float = 0.004
    hook: int = 1
    wall: float = 0.0025


@dataclass
class SpatulaParams:
    blade_len: float = 0.085
    blade_width: float = 0.065
    blade_thickness: float = 0.003
    slots: int = 3
    neck_len: float = 0.05
    handle_len: float = 0.13
    handle_width: float = 0.022
    handle_thickness: float = 0.012
    bend: float = 0.012


@dataclass
class ShapeMods:
    squareness: float = 0.0
    aspect: float = 1.0
    sides: int = 0
    lobes: int = 0
    lobe_depth: float = 0.0
    twist: float = 0.0
    bulge: float = 0.0


@dataclass
class MugParams(ShapeMods):
    radius: float = 0.041
    height: float = 0.095
    taper: float = 1.0
    wall: float = 0.004
    bottom: float = 0.007
    has_handle: int = 1
    handle_size: float = 0.33
    handle_thickness: float = 0.007
    handle_height: float = 0.5


@dataclass
class CupParams(ShapeMods):
    bottom_radius: float = 0.026
    top_radius: float = 0.045
    height: float = 0.06
    belly: float = 0.6
    wall: float = 0.003
    foot: float = 0.006
    has_handle: int = 1
    handle_size: float = 0.3
    handle_thickness: float = 0.005
    saucer: int = 0
    saucer_radius: float = 0.075


@dataclass
class BowlParams(ShapeMods):
    radius: float = 0.075
    height: float = 0.06
    roundness: float = 0.6
    wall: float = 0.004
    foot_radius: float = 0.4
    foot_height: float = 0.007


@dataclass
class PlateParams(ShapeMods):
    radius: float = 0.12
    rim_width: float = 0.25
    rim_height: float = 0.015
    thickness: float = 0.004
    foot_radius: float = 0.55


@dataclass
class GlassParams(ShapeMods):
    radius: float = 0.033
    height: float = 0.11
    taper: float = 1.15
    wall: float = 0.0025
    bottom: float = 0.008


@dataclass
class WineGlassParams(ShapeMods):
    foot_radius: float = 0.035
    stem_height: float = 0.075
    stem_radius: float = 0.0035
    bowl_radius: float = 0.042
    bowl_height: float = 0.1
    rim_ratio: float = 0.75
    wall: float = 0.0022


@dataclass
class PanParams(ShapeMods):
    radius: float = 0.12
    depth: float = 0.045
    flare: float = 0.82
    wall: float = 0.0035
    handle_len: float = 0.19
    handle_width: float = 0.024
    handle_thickness: float = 0.012
    handle_angle: float = 0.12


PARAM_CLASSES = {"fork": ForkParams, "spoon": SpoonParams, "knife": KnifeParams, "ladle": LadleParams,
                 "spatula": SpatulaParams, "mug": MugParams, "cup": CupParams, "bowl": BowlParams,
                 "plate": PlateParams, "glass": GlassParams, "wine_glass": WineGlassParams, "pan": PanParams}
KINDS = list(PARAM_CLASSES)
DISPLAY_NAMES = {"fork": "fork", "spoon": "spoon", "knife": "knife", "ladle": "ladle", "spatula": "spatula",
                 "mug": "mug", "cup": "cup", "bowl": "bowl", "plate": "plate", "glass": "glass",
                 "wine_glass": "wine glass", "pan": "frying pan"}
MESH_VOXEL = {"fork": 0.0007, "spoon": 0.0007, "knife": 0.0006, "ladle": 0.0008, "spatula": 0.0008,
              "mug": 0.001, "cup": 0.0009, "bowl": 0.001, "plate": 0.001, "glass": 0.0008,
              "wine_glass": 0.0008, "pan": 0.0012}
MATERIAL = {"fork": "metal", "spoon": "metal", "knife": "metal", "ladle": "metal", "pan": "metal",
            "spatula": "plastic", "mug": "ceramic", "cup": "ceramic", "bowl": "ceramic", "plate": "ceramic",
            "glass": "glass", "wine_glass": "glass"}

RANGES = {
    "fork": dict(n_tines=(2, 6), handle_len=(0.08, 0.15), handle_width=(0.011, 0.019),
                 neck_len=(0.012, 0.028), neck_width=(0.005, 0.008), head_width=(0.018, 0.034),
                 base_len=(0.010, 0.018), tine_len=(0.022, 0.045), thickness=(0.0025, 0.0035),
                 bend=(0.004, 0.02)),
    "spoon": dict(handle_len=(0.07, 0.14), handle_width=(0.012, 0.020), neck_len=(0.012, 0.025),
                  neck_width=(0.005, 0.008), bowl_len=(0.030, 0.065), bowl_width=(0.022, 0.045),
                  bowl_depth=(0.004, 0.014), thickness=(0.002, 0.003), bend=(0.004, 0.018)),
    "knife": dict(handle_len=(0.08, 0.13), handle_width=(0.014, 0.024), handle_thickness=(0.007, 0.014),
                  blade_len=(0.08, 0.22), blade_width=(0.015, 0.05), blade_thickness=(0.0018, 0.003),
                  tip_round=(0.0, 1.0), serrated=(0, 1)),
    "ladle": dict(bowl_radius=(0.03, 0.055), bowl_depth=(0.5, 1.0), handle_len=(0.18, 0.30),
                  handle_angle=(0.25, 0.7), handle_radius=(0.003, 0.006), hook=(0, 1), wall=(0.002, 0.0035)),
    "spatula": dict(blade_len=(0.06, 0.11), blade_width=(0.045, 0.085), blade_thickness=(0.002, 0.004),
                    slots=(0, 5), neck_len=(0.03, 0.07), handle_len=(0.10, 0.16), handle_width=(0.018, 0.028),
                    handle_thickness=(0.009, 0.016), bend=(0.005, 0.025)),
    "mug": dict(radius=(0.032, 0.05), height=(0.07, 0.125), taper=(0.9, 1.12), wall=(0.003, 0.006),
                bottom=(0.005, 0.01), has_handle=(0, 1), handle_size=(0.25, 0.4),
                handle_thickness=(0.005, 0.009), handle_height=(0.4, 0.6)),
    "cup": dict(bottom_radius=(0.02, 0.03), top_radius=(0.036, 0.052), height=(0.045, 0.075),
                belly=(0.2, 0.9), wall=(0.0025, 0.004), foot=(0.003, 0.008), has_handle=(0, 1),
                handle_size=(0.25, 0.38), handle_thickness=(0.004, 0.0065), saucer=(0, 1),
                saucer_radius=(0.065, 0.085)),
    "bowl": dict(radius=(0.05, 0.11), height=(0.035, 0.09), roundness=(0.2, 1.0), wall=(0.003, 0.0055),
                 foot_radius=(0.3, 0.55), foot_height=(0.004, 0.012)),
    "plate": dict(radius=(0.08, 0.15), rim_width=(0.15, 0.35), rim_height=(0.008, 0.03),
                  thickness=(0.003, 0.006), foot_radius=(0.4, 0.65)),
    "glass": dict(radius=(0.026, 0.042), height=(0.07, 0.16), taper=(0.95, 1.3), wall=(0.002, 0.0035),
                  bottom=(0.004, 0.014)),
    "wine_glass": dict(foot_radius=(0.028, 0.045), stem_height=(0.04, 0.11), stem_radius=(0.0025, 0.005),
                       bowl_radius=(0.03, 0.055), bowl_height=(0.06, 0.13), rim_ratio=(0.55, 1.0),
                       wall=(0.002, 0.003)),
    "pan": dict(radius=(0.09, 0.15), depth=(0.03, 0.065), flare=(0.7, 0.9), wall=(0.003, 0.005),
                handle_len=(0.14, 0.23), handle_width=(0.018, 0.03), handle_thickness=(0.008, 0.016),
                handle_angle=(0.05, 0.25)),
}
CONTAINERS = ["mug", "cup", "bowl", "plate", "glass", "wine_glass", "pan"]
MOD_RANGES = dict(squareness=(0.0, 1.0), aspect=(0.55, 1.0), sides=(0, 8), lobes=(0, 16), lobe_depth=(0.0, 0.25),
                  twist=(-np.pi, np.pi), bulge=(-0.2, 0.25))
MOD_NEUTRAL = dict(squareness=0.0, aspect=1.0, sides=0, lobes=0, lobe_depth=0.0, twist=0.0, bulge=0.0)
for _kind in CONTAINERS:
    RANGES[_kind] = {**MOD_RANGES, **RANGES[_kind]}

BOOL_PROB = {("knife", "serrated"): 0.25, ("ladle", "hook"): 0.6, ("mug", "has_handle"): 0.95,
             ("cup", "has_handle"): 0.9, ("cup", "saucer"): 0.35}


def random_mods(rng):
    m = dict(MOD_NEUTRAL)
    if rng.random() < 0.3:
        m["squareness"] = rng.uniform(0.3, 1.0)
    if rng.random() < 0.3:
        m["aspect"] = rng.uniform(0.55, 0.9)
    if rng.random() < 0.15:
        m["sides"], m["squareness"] = int(rng.integers(3, 9)), 0.0
    if rng.random() < 0.15:
        m["lobes"], m["lobe_depth"] = int(rng.integers(5, 17)), rng.uniform(0.04, 0.2)
    if rng.random() < 0.25 and (m["squareness"] or m["sides"] or m["lobes"]):
        m["twist"] = rng.uniform(-2.5, 2.5)
    if rng.random() < 0.3:
        m["bulge"] = rng.uniform(-0.18, 0.25)
    return m


def random_params(kind, rng, fixed=None):
    fixed = fixed or {}
    mods = {}
    if kind in CONTAINERS:
        mods = dict(MOD_NEUTRAL) if any(k in fixed for k in MOD_NEUTRAL) else random_mods(rng)
    values = {}
    for f in fields(PARAM_CLASSES[kind]):
        if f.name in fixed:
            values[f.name] = fixed[f.name]
        elif f.name in mods:
            values[f.name] = mods[f.name]
        elif (kind, f.name) in BOOL_PROB:
            values[f.name] = int(rng.random() < BOOL_PROB[(kind, f.name)])
        else:
            lo, hi = RANGES[kind][f.name]
            if f.type is int:
                values[f.name] = int(rng.integers(lo, hi + 1))
            else:
                values[f.name] = float(rng.uniform(lo, hi))
    return PARAM_CLASSES[kind](**values)


def fork_outline(x, y, P: ForkParams):
    n = int(round(P.n_tines))
    W = max(P.head_width, n * 0.0042)
    gap = max(0.0018, 0.1 * W)
    tw = (W - (n - 1) * gap) / n
    head_len = P.base_len + P.tine_len
    end = -(P.neck_len + P.handle_len)

    lift = P.bend * (np.maximum(x, 0.0) / head_len) ** 2 + _handle_bend(x, P.neck_len, P.handle_len, 0.7 * P.bend)

    r_end, r_neck = P.handle_width / 2, P.neck_width / 2
    handle = tapered_capsule_2d(x, y, end + r_end, -P.neck_len, r_end, r_neck)
    neck = tapered_capsule_2d(x, y, -P.neck_len, 0.0, r_neck, r_neck)

    half = r_neck + (W / 2 - r_neck) * smoothstep(0.0, P.base_len, x)
    base = np.maximum(np.abs(y) - half, np.maximum(-x, x - P.base_len))

    tines = np.full_like(x, np.inf)
    r_tip = tw * 0.35
    for i in range(n):
        yc = -W / 2 + tw / 2 + i * (tw + gap)
        tines = np.minimum(tines, tapered_capsule_2d(x, y - yc, 0.6 * P.base_len, head_len - r_tip, tw / 2, r_tip))

    d2 = smooth_min(handle, neck, 0.004)
    d2 = smooth_min(d2, base, 0.003)
    return smooth_min(d2, tines, 0.0006), lift


def fork_sdf(p, P: ForkParams):
    d2, lift = fork_outline(p[:, 0], p[:, 1], P)
    t = P.thickness / 2
    return extrude(d2, p[:, 2] - lift, t, round_r=0.6 * t)


def fork_bbox(P: ForkParams):
    W = max(P.head_width, int(round(P.n_tines)) * 0.0042)
    t = P.thickness / 2
    lo = np.array([-(P.neck_len + P.handle_len), -max(W, P.handle_width) / 2, -t])
    hi = np.array([P.base_len + P.tine_len, max(W, P.handle_width) / 2, P.bend + t])
    return lo - PAD, hi + PAD


def spoon_sdf(p, P: SpoonParams):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    end = -(P.neck_len + P.handle_len)
    zc = z - _handle_bend(x, P.neck_len, P.handle_len, P.bend)

    r_end, r_neck = P.handle_width / 2, P.neck_width / 2
    handle = tapered_capsule_2d(x, y, end + r_end, -P.neck_len, r_end, r_neck)
    neck = tapered_capsule_2d(x, y, -P.neck_len, 0.002, r_neck, r_neck * 1.3)
    t = P.thickness / 2
    flat = extrude(smooth_min(handle, neck, 0.004), zc, t, round_r=0.6 * t)

    a, b, c = P.bowl_len / 2, P.bowl_width / 2, P.bowl_depth
    q = np.stack([x - 0.95 * a, y, z], axis=-1)
    shell = np.abs(ellipsoid(q, np.array([a, b, c]))) - t
    return smooth_min(flat, np.maximum(shell, z), 0.003)


def spoon_bbox(P: SpoonParams):
    t = P.thickness / 2
    a, b, c = P.bowl_len / 2, P.bowl_width / 2, P.bowl_depth
    w = max(b, P.handle_width / 2)
    lo = np.array([-(P.neck_len + P.handle_len), -w, -c - t])
    hi = np.array([1.95 * a, w, P.bend + t])
    return lo - PAD, hi + PAD


def knife_outline(x, y, P: KnifeParams):
    r0, r1 = P.handle_width / 2, P.handle_width * 0.42
    handle2d = tapered_capsule_2d(x, y, -P.handle_len + r0, -0.004, r0, r1)
    L, hb = P.blade_len, P.blade_width
    rise = smoothstep(0.5, 1.0, np.clip(x / L, 0.0, 1.0))
    y_bot = -hb / 2 + 0.98 * hb * rise ** 1.3
    mid, half = (hb / 2 + y_bot) / 2, (hb / 2 - y_bot) / 2
    pointed = np.maximum(np.abs(y - mid) - half, np.maximum(-x - 0.002, x - L))
    rc = hb / 2
    rounded = np.minimum(np.maximum(np.abs(y) - rc, np.maximum(-x - 0.002, x - (L - rc))),
                         np.sqrt((x - (L - rc)) ** 2 + y ** 2) - rc)
    blade2d = (1 - P.tip_round) * pointed + P.tip_round * rounded - 0.0005
    if P.serrated:
        period, notch = 0.0045, 0.0013
        edge_y = (1 - P.tip_round) * y_bot + P.tip_round * (-rc)
        xm = np.mod(x, period) - period / 2
        notches = np.hypot(xm, y - edge_y) - notch
        notches = np.where((x > 0.03 * L) & (x < 0.85 * L), notches, np.inf)
        blade2d = np.maximum(blade2d, -notches)
    return handle2d, blade2d


def knife_sdf(p, P: KnifeParams):
    handle2d, blade2d = knife_outline(p[:, 0], p[:, 1], P)
    th, tb = P.handle_thickness / 2, P.blade_thickness / 2
    handle = extrude(handle2d, p[:, 2], th, round_r=0.8 * th)
    blade = extrude(blade2d, p[:, 2], tb, round_r=0.5 * tb)
    return smooth_min(handle, blade, 0.002)


def knife_bbox(P: KnifeParams):
    w = max(P.blade_width, P.handle_width) / 2
    th = P.handle_thickness / 2
    return np.array([-P.handle_len, -w, -th]) - PAD, np.array([P.blade_len, w, th]) + PAD


def _ladle_handle(P: LadleParams):
    Rb, a, L = P.bowl_radius, P.handle_angle, P.handle_len
    h = Rb * P.bowl_depth
    r_rim = np.sqrt(max(Rb * Rb - (Rb - h) ** 2, 1e-12))
    A = np.array([-0.97 * r_rim, 0.0, h])
    B = A + 0.6 * L * np.array([-np.cos(a), 0.0, np.sin(a)])
    C = B + 0.4 * L * np.array([-np.cos(0.25 * a), 0.0, np.sin(0.25 * a)])
    pts = [A, B, C]
    if P.hook:
        pts += [C + np.array([-0.010, 0.0, -0.006]), C + np.array([-0.007, 0.0, -0.022])]
    return h, pts


def ladle_sdf(p, P: LadleParams):
    Rb = P.bowl_radius
    h, pts = _ladle_handle(P)
    dist = np.linalg.norm(p - np.array([0.0, 0.0, Rb]), axis=1)
    shell = np.abs(dist - (Rb - P.wall / 2)) - P.wall / 2
    bowl = np.maximum(shell, p[:, 2] - h)
    handle = polyline3(p, pts) - P.handle_radius
    return smooth_min(bowl, handle, 0.004)


def ladle_bbox(P: LadleParams):
    _, pts = _ladle_handle(P)
    pts = np.array(pts)
    Rb, r = P.bowl_radius, P.handle_radius
    lo = np.minimum(pts.min(0) - r, [-Rb, -Rb, 0.0])
    hi = np.maximum(pts.max(0) + r, [Rb, Rb, Rb])
    return lo - PAD, hi + PAD


def spatula_sdf(p, P: SpatulaParams):
    blade2d, neck2d, handle2d, lift = spatula_outline(p[:, 0], p[:, 1], P)
    zc = p[:, 2] - lift
    tb, th = P.blade_thickness / 2, P.handle_thickness / 2
    blade = extrude(blade2d, zc, tb, round_r=0.4 * tb)
    neck = extrude(neck2d, zc, 1.3 * tb, round_r=0.5 * tb)
    handle = extrude(handle2d, zc, th, round_r=0.8 * th)
    return smooth_min(smooth_min(blade, neck, 0.004), handle, 0.006)


def spatula_outline(x, y, P: SpatulaParams):
    lift = P.bend * smoothstep(0.0, -P.neck_len, x)
    Lb, W = P.blade_len, P.blade_width
    half = W / 2 * (0.72 + 0.28 * smoothstep(0.0, 0.5 * Lb, x))
    blade2d = np.maximum(np.abs(y) - half, np.maximum(-x, x - Lb))
    n = int(round(P.slots))
    if n > 0:
        span = 0.6 * W
        slots = np.full_like(x, np.inf)
        for k in range(n):
            yk = (k - (n - 1) / 2) * span / max(n, 1)
            slots = np.minimum(slots, tapered_capsule_2d(x, y - yk, 0.25 * Lb, 0.8 * Lb, 0.0025, 0.0025))
        blade2d = np.maximum(blade2d, -slots)
    neck2d = tapered_capsule_2d(x, y, -P.neck_len, 0.15 * Lb, 0.005, 0.012)
    hr = P.handle_width / 2
    handle2d = tapered_capsule_2d(x, y, -(P.neck_len + P.handle_len) + hr, -P.neck_len, hr, 0.75 * hr)
    return blade2d, neck2d, handle2d, lift


def spatula_bbox(P: SpatulaParams):
    w = max(P.blade_width, P.handle_width) / 2
    th = P.handle_thickness / 2
    lo = np.array([-(P.neck_len + P.handle_len), -w, -th])
    hi = np.array([P.blade_len, w, P.bend + th])
    return lo - PAD, hi + PAD


def cross_radius(x, y, z, P, height):
    t = np.clip(z / max(height, 1e-6), 0.0, 1.0)
    if P.twist:
        a = P.twist * t
        c, s = np.cos(a), np.sin(a)
        x, y = c * x + s * y, -s * x + c * y
    y = y / P.aspect
    if P.sides >= 3:
        n = int(P.sides)
        beta = np.mod(np.arctan2(y, x) + np.pi / n, 2 * np.pi / n) - np.pi / n
        r = np.hypot(x, y) * np.cos(beta) / np.cos(np.pi / n)
    elif P.squareness > 0:
        p = 2.0 + 6.0 * P.squareness
        r = (np.abs(x) ** p + np.abs(y) ** p) ** (1 / p)
    else:
        r = np.hypot(x, y)
    if P.lobes >= 3 and P.lobe_depth > 0:
        r = r / (1 + P.lobe_depth * np.cos(P.lobes * np.arctan2(y, x)))
    if P.bulge:
        r = r / (1 + P.bulge * np.sin(np.pi * t))
    return r


def _spread(P):
    f = (1 + P.lobe_depth if P.lobes >= 3 else 1.0) * (1 + max(P.bulge, 0.0))
    if P.twist and P.squareness > 0 and P.sides < 3:
        f *= 1.42
    return f


def _outer_x(body_fn, z, reach, direction=1.0, n=800):
    xs = np.linspace(0.0, reach, n)
    pts = np.stack([direction * xs, np.zeros(n), np.full(n, z)], -1)
    solid = np.flatnonzero(body_fn(pts) <= 0)
    return float(xs[solid[-1]]) if len(solid) else 0.0


def _side_handle(x, y, z, r, cx, zc, ring_r, thickness, inner_radius):
    ring = np.hypot(np.hypot(x - cx, z - zc) - ring_r, y) - thickness / 2
    return np.maximum(ring, inner_radius - r)


def _straight_wall(r, z, radius, taper, height, wall, bottom):
    r0, r1 = radius - wall / 2, radius * taper - wall / 2
    side = seg2(r, z, r0, bottom / 2, r1, height - wall / 2) - wall / 2
    base = seg2(r, z, 0.0, bottom / 2, r0, bottom / 2) - bottom / 2
    t = np.clip((z - bottom / 2) / (height - wall / 2 - bottom / 2), 0.0, 1.0)
    inner = r0 + (r1 - r0) * t - wall / 2
    return smooth_min(side, base, 0.003), inner


def _mug_body(p, P: MugParams):
    r = cross_radius(p[:, 0], p[:, 1], p[:, 2], P, P.height)
    body, inner = _straight_wall(r, p[:, 2], P.radius, P.taper, P.height, P.wall, P.bottom)
    return body, inner, r


def _mug_handle_place(P: MugParams):
    zc, ring_r = P.handle_height * P.height, P.handle_size * P.height
    reach = 2 * P.radius * max(1.0, P.taper) * _spread(P)
    return zc, ring_r, _outer_x(lambda q: _mug_body(q, P)[0], zc, reach)


def mug_sdf(p, P: MugParams):
    body, inner, r = _mug_body(p, P)
    if not P.has_handle:
        return body
    zc, ring_r, wall_out = _mug_handle_place(P)
    handle = _side_handle(p[:, 0], p[:, 1], p[:, 2], r, wall_out + 0.3 * ring_r, zc, ring_r,
                          P.handle_thickness, inner)
    return smooth_min(body, handle, 0.003)


def mug_bbox(P: MugParams):
    R = P.radius * max(1.0, P.taper) * _spread(P)
    x_hi = R
    if P.has_handle:
        _, ring_r, wall_out = _mug_handle_place(P)
        x_hi = max(R, wall_out + 1.3 * ring_r + P.handle_thickness / 2)
    return np.array([-R, -R, 0.0]) - PAD, np.array([x_hi, R, P.height]) + PAD


SAUCER_THICK, SAUCER_FOOT = 0.004, 0.003
SAUCER_LIFT = SAUCER_THICK + SAUCER_FOOT


def _cup_profile(P: CupParams):
    w = P.wall
    z0 = P.foot + w / 2
    s = np.linspace(0.0, 1.0, 12)
    rs = P.bottom_radius + (P.top_radius - w / 2 - P.bottom_radius) * (1 - (1 - s) ** (1 + 2 * P.belly))
    zs = z0 + (P.height - w / 2 - z0) * s
    return rs, zs


def _cup_body(q, P: CupParams):
    z = q[:, 2]
    r = cross_radius(q[:, 0], q[:, 1], z, P, P.height)
    rs, zs = _cup_profile(P)
    bowl = polyline2(r, z, [(0.0, zs[0])] + list(zip(rs, zs))) - P.wall / 2
    rf = 0.8 * P.bottom_radius
    foot = seg2(r, z, rf, P.wall / 2, rf, P.foot + P.wall / 2) - P.wall / 2
    return smooth_min(bowl, foot, 0.002), r, rs, zs


def _cup_handle_place(P: CupParams):
    zh = P.foot + 0.6 * (P.height - P.foot)
    ring_r = 0.8 * P.handle_size * P.height
    return zh, ring_r, _outer_x(lambda q: _cup_body(q, P)[0], zh, 2 * P.top_radius * _spread(P))


def cup_sdf(p, P: CupParams):
    q = p - np.array([0.0, 0.0, SAUCER_LIFT if P.saucer else 0.0])
    d, r, rs, zs = _cup_body(q, P)
    if P.has_handle:
        zh, ring_r, r_out = _cup_handle_place(P)
        inner = np.where(q[:, 2] > P.foot + P.wall, np.interp(q[:, 2], zs, rs) - P.wall / 2, -1.0)
        handle = _side_handle(q[:, 0], q[:, 1], q[:, 2], r, r_out + 0.25 * ring_r, zh, ring_r,
                              P.handle_thickness, inner)
        d = smooth_min(d, handle, 0.002)
    if P.saucer:
        z = p[:, 2]
        rs_ = cross_radius(p[:, 0], p[:, 1], np.zeros(len(p)), P, P.height)
        Rs, st, fh = P.saucer_radius, SAUCER_THICK, SAUCER_FOOT
        prof = [(0.0, fh + st / 2), (0.4 * Rs, fh + st / 2), (0.85 * Rs, fh + st / 2 + 0.006),
                (Rs - st / 2, fh + st / 2 + 0.011)]
        saucer = smooth_min(polyline2(rs_, z, prof) - st / 2,
                            seg2(rs_, z, 0.35 * Rs, st / 2, 0.35 * Rs, fh + st / 2) - st / 2, 0.002)
        d = np.minimum(d, saucer)
    return d


def cup_bbox(P: CupParams):
    R = max(P.top_radius, P.saucer_radius if P.saucer else 0.0) * _spread(P)
    x_hi = R
    if P.has_handle:
        _, ring_r, r_out = _cup_handle_place(P)
        x_hi = max(R, r_out + 1.25 * ring_r + P.handle_thickness / 2)
    top = P.height + (SAUCER_LIFT if P.saucer else 0.0)
    return np.array([-R, -R, 0.0]) - PAD, np.array([x_hi, R, top]) + PAD


def _bowl_profile(P: BowlParams):
    w = P.wall
    z0 = P.foot_height + w / 2
    R, H = P.radius - w / 2, P.height - w / 2
    rf = P.foot_radius * P.radius
    s = np.linspace(0.0, 1.0, 14)
    phi = s * np.pi / 2
    k = P.roundness
    rs = k * R * np.sin(phi) + (1 - k) * (rf + (R - rf) * s)
    zs = k * (z0 + (H - z0) * (1 - np.cos(phi))) + (1 - k) * (z0 + (H - z0) * s)
    return rs, zs


def bowl_sdf(p, P: BowlParams):
    z = p[:, 2]
    r = cross_radius(p[:, 0], p[:, 1], z, P, P.height)
    rs, zs = _bowl_profile(P)
    shell = polyline2(r, z, [(0.0, zs[0])] + list(zip(rs, zs))) - P.wall / 2
    rf = P.foot_radius * P.radius
    foot_top = np.interp(rf, rs, zs)
    foot = seg2(r, z, rf, P.wall / 2, rf, foot_top) - P.wall / 2
    return smooth_min(shell, foot, 0.003)


def bowl_bbox(P: BowlParams):
    R = P.radius * _spread(P)
    return np.array([-R, -R, 0.0]) - PAD, np.array([R, R, P.height]) + PAD


def plate_sdf(p, P: PlateParams):
    z = p[:, 2]
    t, fh = P.thickness, 0.004
    r = cross_radius(p[:, 0], p[:, 1], z, P, fh + t + P.rim_height)
    z0 = fh + t / 2
    R, Ri = P.radius - t / 2, P.radius * (1 - P.rim_width)
    prof = [(0.0, z0), (Ri, z0), (Ri + 0.55 * (R - Ri), z0 + 0.35 * P.rim_height), (R, z0 + P.rim_height)]
    shell = polyline2(r, z, prof) - t / 2
    rf = P.foot_radius * Ri
    foot = seg2(r, z, rf, t / 2, rf, z0) - t / 2
    return smooth_min(shell, foot, 0.003)


def plate_bbox(P: PlateParams):
    R = P.radius * _spread(P)
    return np.array([-R, -R, 0.0]) - PAD, np.array([R, R, 0.004 + P.thickness + P.rim_height]) + PAD


def glass_sdf(p, P: GlassParams):
    r = cross_radius(p[:, 0], p[:, 1], p[:, 2], P, P.height)
    body, _ = _straight_wall(r, p[:, 2], P.radius, P.taper, P.height, P.wall, P.bottom)
    return body


def glass_bbox(P: GlassParams):
    R = P.radius * max(1.0, P.taper) * _spread(P)
    return np.array([-R, -R, 0.0]) - PAD, np.array([R, R, P.height]) + PAD


WINE_FOOT_THICK = 0.0035


def wine_glass_sdf(p, P: WineGlassParams):
    z = p[:, 2]
    ft = WINE_FOOT_THICK
    r = cross_radius(p[:, 0], p[:, 1], z, P, ft + P.stem_height + P.bowl_height)
    foot = seg2(r, z, 0.0, ft / 2 + 0.0025, P.foot_radius - ft / 2, ft / 2) - ft / 2
    zb = ft + P.stem_height
    stem = seg2(r, z, 0.0, ft, 0.0, zb) - P.stem_radius
    s = np.linspace(0.0, 1.0, 12)
    Rm = P.bowl_radius - P.wall / 2
    Rr, sw = Rm * P.rim_ratio, 0.42
    upper = Rm - (Rm - Rr) * (np.maximum(s - sw, 0.0) / (1 - sw)) ** 1.6
    rs = np.where(s <= sw, Rm * np.sin(np.pi / 2 * np.minimum(s / sw, 1.0)), upper)
    zs = zb + (P.bowl_height - P.wall / 2) * s
    bowl = polyline2(r, z, list(zip(rs, zs))) - P.wall / 2
    return smooth_min(smooth_min(foot, stem, 0.006), bowl, 0.008)


def wine_glass_bbox(P: WineGlassParams):
    R = max(P.foot_radius, P.bowl_radius) * _spread(P)
    top = WINE_FOOT_THICK + P.stem_height + P.bowl_height
    return np.array([-R, -R, 0.0]) - PAD, np.array([R, R, top]) + PAD


def _pan_body(p, P: PanParams):
    z = p[:, 2]
    r = cross_radius(p[:, 0], p[:, 1], z, P, P.depth)
    w, R, D = P.wall, P.radius, P.depth
    Rb = R * P.flare
    prof = [(0.0, w / 2), (Rb - 0.01, w / 2), (Rb + 0.35 * (R - Rb), 0.12 * D + w / 2), (R - w / 2, D - w / 2)]
    return polyline2(r, z, prof) - w / 2


def _pan_handle_frame(P: PanParams):
    a = P.handle_angle
    z0 = P.depth - P.handle_thickness / 2 - 0.004
    wall = _outer_x(lambda q: _pan_body(q, P), z0, 2 * P.radius * _spread(P), direction=-1.0)
    start = np.array([-(wall - 0.006), 0.0, z0])
    along = np.array([-np.cos(a), 0.0, np.sin(a)])
    up = np.array([np.sin(a), 0.0, np.cos(a)])
    return start, along, up


def pan_sdf(p, P: PanParams):
    body = _pan_body(p, P)
    start, along, up = _pan_handle_frame(P)
    q = p - start
    u, v, n = q @ along, p[:, 1], q @ up
    L = P.handle_len
    handle = rounded_box(u - L / 2, v, n, L / 2, P.handle_width / 2, P.handle_thickness / 2, 0.4 * P.handle_thickness)
    hole = np.hypot(u - (L - 0.02), v) - 0.0055
    return smooth_min(body, np.maximum(handle, -hole), 0.006)


def pan_bbox(P: PanParams):
    start, along, up = _pan_handle_frame(P)
    tip = start + P.handle_len * along
    R = P.radius * _spread(P)
    lo = np.array([min(-R, tip[0] - P.handle_thickness), -max(R, P.handle_width), 0.0])
    hi = np.array([R, max(R, P.handle_width), max(P.depth, tip[2] + P.handle_thickness)])
    return lo - PAD, hi + PAD


WARP_FRAME = {"mug": lambda P: (0.0, P.height),
              "cup": lambda P: (SAUCER_LIFT if P.saucer else 0.0, P.height),
              "bowl": lambda P: (0.0, P.height),
              "plate": lambda P: (0.0, 0.004 + P.thickness + P.rim_height),
              "glass": lambda P: (0.0, P.height),
              "wine_glass": lambda P: (0.0, WINE_FOOT_THICK + P.stem_height + P.bowl_height),
              "pan": lambda P: (0.0, P.depth)}


def warp_radius(kind, params, p):
    if kind not in WARP_FRAME:
        return np.zeros(len(p))
    z0, height = WARP_FRAME[kind](params)
    return cross_radius(p[:, 0], p[:, 1], p[:, 2] - z0, params, height)


SDF = {"fork": fork_sdf, "spoon": spoon_sdf, "knife": knife_sdf, "ladle": ladle_sdf, "spatula": spatula_sdf,
       "mug": mug_sdf, "cup": cup_sdf, "bowl": bowl_sdf, "plate": plate_sdf, "glass": glass_sdf,
       "wine_glass": wine_glass_sdf, "pan": pan_sdf}
BBOX = {"fork": fork_bbox, "spoon": spoon_bbox, "knife": knife_bbox, "ladle": ladle_bbox,
        "spatula": spatula_bbox, "mug": mug_bbox, "cup": cup_bbox, "bowl": bowl_bbox, "plate": plate_bbox,
        "glass": glass_bbox, "wine_glass": wine_glass_bbox, "pan": pan_bbox}


def shape_sdf(kind, params):
    return lambda p: SDF[kind](p, params)


def shape_bbox(kind, params):
    return BBOX[kind](params)


def params_dict(params):
    return asdict(params)


LENGTH_FIELDS = {"fork": ["handle_len", "neck_len", "base_len", "tine_len"],
                 "spoon": ["handle_len", "neck_len", "bowl_len"],
                 "knife": ["handle_len", "blade_len"],
                 "ladle": ["handle_len"],
                 "spatula": ["blade_len", "neck_len", "handle_len"],
                 "pan": ["handle_len"]}
SIZE_FIELDS = {"mug": ["radius", "height"], "cup": ["bottom_radius", "top_radius", "height", "saucer_radius"],
               "glass": ["radius", "height"], "wine_glass": ["bowl_radius", "bowl_height"],
               "bowl": ["radius", "height"], "ladle": ["bowl_radius"], "pan": ["radius", "depth"],
               "plate": ["radius"]}
FLAT_KINDS = {"fork", "spoon", "knife", "ladle", "spatula"}


def total_length(kind, params):
    lo, hi = BBOX[kind](params)
    return float(hi[0] - lo[0] - 2 * PAD)


def total_height(kind, params):
    lo, hi = BBOX[kind](params)
    return float(hi[2] - lo[2] - 2 * PAD)


def scale_to_length(kind, params, target):
    names = LENGTH_FIELDS.get(kind)
    if not names:
        return params
    factor = target / total_length(kind, params)
    for name in names:
        setattr(params, name, getattr(params, name) * factor)
    return params


def set_height(kind, params, target):
    if kind in ("mug", "glass", "bowl"):
        params.height = target
    elif kind == "cup":
        params.height = target - (SAUCER_LIFT if params.saucer else 0.0)
    elif kind == "wine_glass":
        k = (target - WINE_FOOT_THICK) / (params.stem_height + params.bowl_height)
        params.stem_height *= k
        params.bowl_height *= k
    elif kind == "pan":
        params.depth = target
    return params


def set_diameter(kind, params, target):
    field = {"mug": "radius", "glass": "radius", "bowl": "radius", "plate": "radius", "pan": "radius",
             "cup": "top_radius", "wine_glass": "bowl_radius", "ladle": "bowl_radius"}.get(kind)
    if field:
        setattr(params, field, target / 2)
    return params


def volume(kind, P):
    frustum = lambda h, a, b: np.pi * h / 3 * (a * a + a * b + b * b)
    if kind in ("mug", "glass"):
        return frustum(P.height - P.bottom, P.radius - P.wall, P.radius * P.taper - P.wall)
    if kind == "cup":
        return frustum(P.height - P.foot, P.bottom_radius, P.top_radius - P.wall) * (1 + 0.25 * P.belly)
    if kind == "bowl":
        rf = P.foot_radius * P.radius
        cone = frustum(P.height, rf, P.radius)
        return P.roundness * (2 / 3) * np.pi * P.radius ** 2 * P.height + (1 - P.roundness) * cone
    if kind == "wine_glass":
        return 0.55 * np.pi * P.bowl_radius ** 2 * P.bowl_height
    if kind == "ladle":
        h = P.bowl_radius * P.bowl_depth
        return np.pi * h * h * (3 * P.bowl_radius - h) / 3
    if kind == "pan":
        return frustum(P.depth, P.radius * P.flare, P.radius)
    return None


def set_capacity(kind, params, target_m3):
    current = volume(kind, params)
    if not current or kind not in SIZE_FIELDS:
        return params
    k = (target_m3 / current) ** (1 / 3)
    for name in SIZE_FIELDS[kind]:
        setattr(params, name, getattr(params, name) * k)
    return params
