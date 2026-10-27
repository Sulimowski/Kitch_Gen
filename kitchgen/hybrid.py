from types import SimpleNamespace

import numpy as np

from .shapes import (BBOX, CONTAINERS, MOD_NEUTRAL, PAD, SAUCER_FOOT, SAUCER_LIFT, SAUCER_THICK, WINE_FOOT_THICK,
                     _bowl_profile, _cup_profile, _handle_bend, _outer_x, _side_handle, _spread, cross_radius,
                     extrude, fork_outline, knife_outline, polyline2, rounded_box, seg2, smooth_min, smoothstep,
                     spatula_outline, tapered_capsule_2d)

FLAT = ["fork", "spoon", "knife", "spatula"]
KEEP = 0.35
N_PROFILE = 28


def family(kind):
    return "vessel" if kind in CONTAINERS else ("flat" if kind in FLAT else None)


def can_cross(kind_a, kind_b):
    return kind_a != kind_b and family(kind_a) is not None and family(kind_a) == family(kind_b)


def _mix(a, b, w):
    return (1 - w) * a + w * b


def tapered_polyline2(px, py, pts, ht):
    d = np.full_like(px, np.inf)
    for i in range(len(pts) - 1):
        (ax, ay), (bx, by) = pts[i], pts[i + 1]
        dx, dy = bx - ax, by - ay
        h = np.clip(((px - ax) * dx + (py - ay) * dy) / max(dx * dx + dy * dy, 1e-12), 0.0, 1.0)
        d = np.minimum(d, np.hypot(px - ax - h * dx, py - ay - h * dy) - (ht[i] + h * (ht[i + 1] - ht[i])))
    return d


def _stroke(points, ht):
    return np.asarray(points, float), np.asarray(ht, float)


def vessel_of(kind, P):
    V = dict(mods=SimpleNamespace(**{k: getattr(P, k) for k in MOD_NEUTRAL}), extras=[], handle=None,
             pan_handle=None, saucer=None, lift=0.0)
    if kind in ("mug", "glass"):
        w, b, H = P.wall, P.bottom, P.height
        r0, r1 = P.radius - w / 2, P.radius * P.taper - w / 2
        k = 0.06
        V["main"] = _stroke([(0.0, b / 2), (r0, b / 2), (r0 + k * (r1 - r0), b / 2 + k * (H - w / 2 - b / 2)),
                             (r1, H - w / 2)], [b / 2, b / 2, w / 2, w / 2])
        V["height"] = H
        if kind == "mug" and P.has_handle:
            V["handle"] = dict(zc=P.handle_height * H, ring_r=P.handle_size * H, thickness=P.handle_thickness,
                               offset=0.3)
    elif kind == "cup":
        rs, zs = _cup_profile(P)
        w = P.wall
        V["main"] = _stroke([(0.0, zs[0])] + list(zip(rs, zs)), [w / 2] * (len(rs) + 1))
        rf = 0.8 * P.bottom_radius
        V["extras"].append(_stroke([(rf, w / 2), (rf, P.foot + w / 2)], [w / 2, w / 2]))
        V["height"] = P.height
        if P.has_handle:
            V["handle"] = dict(zc=P.foot + 0.6 * (P.height - P.foot), ring_r=0.8 * P.handle_size * P.height,
                               thickness=P.handle_thickness, offset=0.25)
        if P.saucer:
            V["saucer"], V["lift"] = P.saucer_radius, SAUCER_LIFT
    elif kind == "bowl":
        rs, zs = _bowl_profile(P)
        w, rf = P.wall, P.foot_radius * P.radius
        V["main"] = _stroke([(0.0, zs[0])] + list(zip(rs, zs)), [w / 2] * (len(rs) + 1))
        V["extras"].append(_stroke([(rf, w / 2), (rf, float(np.interp(rf, rs, zs)))], [w / 2, w / 2]))
        V["height"] = P.height
    elif kind == "plate":
        t, fh = P.thickness, 0.004
        z0 = fh + t / 2
        R, Ri = P.radius - t / 2, P.radius * (1 - P.rim_width)
        V["main"] = _stroke([(0.0, z0), (Ri, z0), (Ri + 0.55 * (R - Ri), z0 + 0.35 * P.rim_height),
                             (R, z0 + P.rim_height)], [t / 2] * 4)
        rf = P.foot_radius * Ri
        V["extras"].append(_stroke([(rf, t / 2), (rf, z0)], [t / 2, t / 2]))
        V["height"] = fh + t + P.rim_height
    elif kind == "wine_glass":
        ft = WINE_FOOT_THICK
        zb = ft + P.stem_height
        s = np.linspace(0.0, 1.0, 12)
        Rm = P.bowl_radius - P.wall / 2
        Rr, sw = Rm * P.rim_ratio, 0.42
        upper = Rm - (Rm - Rr) * (np.maximum(s - sw, 0.0) / (1 - sw)) ** 1.6
        rs = np.where(s <= sw, Rm * np.sin(np.pi / 2 * np.minimum(s / sw, 1.0)), upper)
        zs = zb + (P.bowl_height - P.wall / 2) * s
        V["main"] = _stroke([(0.0, ft)] + list(zip(rs, zs)), [P.stem_radius] * 2 + [P.wall / 2] * (len(rs) - 1))
        V["extras"].append(_stroke([(0.0, ft / 2 + 0.0025), (P.foot_radius - ft / 2, ft / 2)], [ft / 2, ft / 2]))
        V["height"] = zb + P.bowl_height
    elif kind == "pan":
        w, R, D = P.wall, P.radius, P.depth
        Rb = R * P.flare
        V["main"] = _stroke([(0.0, w / 2), (Rb - 0.01, w / 2), (Rb + 0.35 * (R - Rb), 0.12 * D + w / 2),
                             (R - w / 2, D - w / 2)], [w / 2] * 4)
        V["height"] = D
        V["pan_handle"] = dict(angle=P.handle_angle, length=P.handle_len, width=P.handle_width,
                               thickness=P.handle_thickness)
    return V


def _resample(stroke, n=N_PROFILE):
    pts, ht = stroke
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    s /= max(s[-1], 1e-12)
    u = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(u, s, pts[:, 0]), np.interp(u, s, pts[:, 1])], -1), np.interp(u, s, ht)


def mix_vessels(VA, VB, w):
    (pa, ha), (pb, hb) = _resample(VA["main"]), _resample(VB["main"])
    V = dict(main=(_mix(pa, pb, w), _mix(ha, hb, w)), height=_mix(VA["height"], VB["height"], w))
    V["extras"] = (VA["extras"] if 1 - w >= KEEP else []) + (VB["extras"] if w >= KEEP else [])
    ma, mb = VA["mods"], VB["mods"]
    mods = {k: _mix(getattr(ma, k), getattr(mb, k), w) for k in ("squareness", "aspect", "lobe_depth", "twist", "bulge")}
    for k in ("sides", "lobes"):
        va, vb = getattr(ma, k), getattr(mb, k)
        if va and 1 - w >= KEEP and (not vb or w < 0.5):
            mods[k] = va
        else:
            mods[k] = vb if vb and w >= KEEP else 0
    V["mods"] = SimpleNamespace(**mods)
    for key in ("handle", "pan_handle"):
        A, B = VA[key], VB[key]
        if A and B:
            V[key] = {k: _mix(A[k], B[k], w) for k in A}
        else:
            V[key] = A if A and 1 - w >= KEEP else (B if B and w >= KEEP else None)
    sa, sb = VA["saucer"], VB["saucer"]
    if sa and sb:
        V["saucer"] = _mix(sa, sb, w)
    else:
        V["saucer"] = sa if sa and 1 - w >= KEEP else (sb if sb and w >= KEEP else None)
    V["lift"] = SAUCER_LIFT if V["saucer"] else 0.0
    return V


def _vessel_body(q, V):
    z = q[:, 2]
    r = cross_radius(q[:, 0], q[:, 1], z, V["mods"], V["height"])
    d = tapered_polyline2(r, z, *V["main"])
    for pts, ht in V["extras"]:
        d = smooth_min(d, tapered_polyline2(r, z, pts, ht), 0.002)
    return d, r


def _cavity_radius(z, V):
    pts, ht = V["main"]
    zs = np.maximum.accumulate(pts[1:, 1])
    inner = np.interp(z, zs, pts[1:, 0] - ht[1:])
    return np.where(z > pts[0, 1] + ht[0], inner, -1.0)


def _reach(V):
    strokes = [V["main"]] + V["extras"]
    return max(float(np.max(p[:, 0] + h)) for p, h in strokes) * _spread(V["mods"])


def _pan_frame(V):
    ph = V["pan_handle"]
    top = float(np.max(V["main"][0][:, 1]))
    z0 = top - ph["thickness"] / 2 - 0.004
    wall = _outer_x(lambda s: _vessel_body(s, V)[0], z0, 2 * _reach(V), direction=-1.0)
    a = ph["angle"]
    start = np.array([-(wall - 0.006), 0.0, z0])
    return start, np.array([-np.cos(a), 0.0, np.sin(a)]), np.array([np.sin(a), 0.0, np.cos(a)])


def vessel_sdf(V):
    def fn(p):
        q = p - np.array([0.0, 0.0, V["lift"]])
        d, r = _vessel_body(q, V)
        if V["handle"]:
            h = V["handle"]
            wall = _outer_x(lambda s: _vessel_body(s, V)[0], h["zc"], 2 * _reach(V))
            handle = _side_handle(q[:, 0], q[:, 1], q[:, 2], r, wall + h["offset"] * h["ring_r"], h["zc"],
                                  h["ring_r"], h["thickness"], _cavity_radius(q[:, 2], V))
            d = smooth_min(d, handle, 0.003)
        if V["pan_handle"]:
            ph = V["pan_handle"]
            start, along, up = _pan_frame(V)
            s = q - start
            u, n = s @ along, s @ up
            L = ph["length"]
            box = rounded_box(u - L / 2, q[:, 1], n, L / 2, ph["width"] / 2, ph["thickness"] / 2, 0.4 * ph["thickness"])
            hole = np.hypot(u - (L - 0.02), q[:, 1]) - 0.0055
            d = smooth_min(d, np.maximum(box, -hole), 0.006)
        if V["saucer"]:
            z = p[:, 2]
            rs_ = cross_radius(p[:, 0], p[:, 1], np.zeros(len(p)), V["mods"], V["height"])
            Rs, st, fh = V["saucer"], SAUCER_THICK, SAUCER_FOOT
            prof = [(0.0, fh + st / 2), (0.4 * Rs, fh + st / 2), (0.85 * Rs, fh + st / 2 + 0.006),
                    (Rs - st / 2, fh + st / 2 + 0.011)]
            saucer = smooth_min(polyline2(rs_, z, prof) - st / 2,
                                seg2(rs_, z, 0.35 * Rs, st / 2, 0.35 * Rs, fh + st / 2) - st / 2, 0.002)
            d = np.minimum(d, saucer)
        return d
    return fn


def vessel_bbox(V):
    R = _reach(V)
    strokes = [V["main"]] + V["extras"]
    top = max(float(np.max(p[:, 1] + h)) for p, h in strokes) + V["lift"]
    lo, hi = np.array([-R, -R, 0.0]), np.array([R, R, top])
    if V["saucer"]:
        Rs = V["saucer"] * _spread(V["mods"])
        lo[:2], hi[:2] = np.minimum(lo[:2], -Rs), np.maximum(hi[:2], Rs)
    if V["handle"]:
        h = V["handle"]
        wall = _outer_x(lambda s: _vessel_body(s, V)[0], h["zc"], 2 * R)
        hi[0] = max(hi[0], wall + (1 + h["offset"]) * h["ring_r"] + h["thickness"])
    if V["pan_handle"]:
        ph = V["pan_handle"]
        start, along, _ = _pan_frame(V)
        tip = start + ph["length"] * along
        lo[0] = min(lo[0], tip[0] - ph["thickness"])
        hi[2] = max(hi[2], tip[2] + ph["thickness"])
        lo[1], hi[1] = min(lo[1], -ph["width"]), max(hi[1], ph["width"])
    return lo - PAD, hi + PAD


def flat_of(kind, P):
    if kind == "fork":
        def parts(x, y):
            d2, lift = fork_outline(x, y, P)
            return d2, np.full_like(x, P.thickness / 2), lift
    elif kind == "spoon":
        def parts(x, y):
            r_end, r_neck = P.handle_width / 2, P.neck_width / 2
            end = -(P.neck_len + P.handle_len)
            handle = tapered_capsule_2d(x, y, end + r_end, -P.neck_len, r_end, r_neck)
            neck = tapered_capsule_2d(x, y, -P.neck_len, 0.002, r_neck, r_neck * 1.3)
            a, b = P.bowl_len / 2, P.bowl_width / 2
            u, v = (x - 0.95 * a) / a, y / b
            k0, k1 = np.hypot(u, v), np.hypot(u / a, v / b)
            bowl = k0 * (k0 - 1) / np.maximum(k1, 1e-12)
            d2 = smooth_min(smooth_min(handle, neck, 0.004), bowl, 0.003)
            dish = -P.bowl_depth * np.clip(1 - u * u - v * v, 0.0, 1.0)
            return d2, np.full_like(x, P.thickness / 2), _handle_bend(x, P.neck_len, P.handle_len, P.bend) + dish
    elif kind == "knife":
        def parts(x, y):
            handle2d, blade2d = knife_outline(x, y, P)
            s = smoothstep(-0.006, 0.002, x)
            return (smooth_min(handle2d, blade2d, 0.002),
                    (1 - s) * P.handle_thickness / 2 + s * P.blade_thickness / 2, np.zeros_like(x))
    else:
        def parts(x, y):
            blade2d, neck2d, handle2d, lift = spatula_outline(x, y, P)
            s = smoothstep(0.0, -P.neck_len, x)
            return (smooth_min(smooth_min(blade2d, neck2d, 0.004), handle2d, 0.006),
                    (1 - s) * P.blade_thickness / 2 + s * P.handle_thickness / 2, lift)
    return parts


def flat_cross_sdf(kind_a, Pa, kind_b, Pb, w):
    fa, fb = flat_of(kind_a, Pa), flat_of(kind_b, Pb)

    def fn(p):
        da, ta, la = fa(p[:, 0], p[:, 1])
        db, tb, lb = fb(p[:, 0], p[:, 1])
        t = _mix(ta, tb, w)
        return extrude(_mix(da, db, w), p[:, 2] - _mix(la, lb, w), t, round_r=0.5 * t)
    return fn


def cross(kind_a, Pa, kind_b, Pb, w):
    if family(kind_a) == "vessel":
        V = mix_vessels(vessel_of(kind_a, Pa), vessel_of(kind_b, Pb), w)
        lo, hi = vessel_bbox(V)
        return vessel_sdf(V), lo, hi
    lo_a, hi_a = BBOX[kind_a](Pa)
    lo_b, hi_b = BBOX[kind_b](Pb)
    return flat_cross_sdf(kind_a, Pa, kind_b, Pb, w), np.minimum(lo_a, lo_b), np.maximum(hi_a, hi_b)
