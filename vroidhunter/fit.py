"""Fit a VRM avatar onto the hunter skeleton.

The game animates armor with its own hunter skeleton, so the avatar's mesh has
to be reshaped into the hunter's rest pose:

1. Rotate the avatar so its up / left directions match the hunter's.
2. Scale it uniformly so ankle-to-head height equals the hunter's.
3. Move every limb segment so its joints land on the matching hunter joints
   (rotating and stretching the segment along its length as needed).
4. Re-target skin weights from VRM bones to hunter bone function ids.
"""

from dataclasses import dataclass

import numpy as np

# Hunter bone function ids (community tables, see THIRD_PARTY_NOTICES.md).
ROOT, WAIST, CHEST, NECK, HEAD = 0, 1, 2, 3, 4
PELVIS = 13

LEFT = dict(shoulder=5, upper=(6, 80), fore=7, hand=8, fore_twist=81,
            thigh=14, calf=15, foot=16, toe=17,
            thumb=(31, 32, 33), index=(34, 35, 36), middle=(37, 38, 39),
            ring=(41, 42, 43), little=(44, 45, 46))
RIGHT = dict(shoulder=9, upper=(10, 82), fore=11, hand=12, fore_twist=83,
             thigh=18, calf=19, foot=20, toe=21,
             thumb=(48, 49, 50), index=(51, 52, 53), middle=(54, 55, 56),
             ring=(58, 59, 60), little=(61, 62, 63))

FINGERS = (("Thumb", "thumb", ("1", "2", "3")),
           ("Index", "index", ("Proximal", "Intermediate", "Distal")),
           ("Middle", "middle", ("Proximal", "Intermediate", "Distal")),
           ("Ring", "ring", ("Proximal", "Intermediate", "Distal")),
           ("Little", "little", ("Proximal", "Intermediate", "Distal")))

TORSO = ("hips", "spine", "chest", "upperChest")


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else None


def rotation_between(u, v):
    """Smallest rotation taking direction u to direction v."""
    u, v = _unit(u), _unit(v)
    if u is None or v is None:
        return np.eye(3)
    c = float(np.dot(u, v))
    axis = np.cross(u, v)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        perp = np.cross(u, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(u, [0.0, 1.0, 0.0])
        perp = _unit(perp)
        return 2 * np.outer(perp, perp) - np.eye(3)
    k = axis / s
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * kx + (1 - c) * (kx @ kx)


def frame(primary, secondary):
    """Orthonormal basis (columns) from a primary and a secondary direction."""
    a = _unit(primary)
    if a is None:
        return None
    b = secondary - a * np.dot(a, secondary)
    b = _unit(b)
    if b is None:
        return None
    return np.column_stack([a, b, np.cross(a, b)])


@dataclass
class Affine:
    """x -> target + A @ (x - source)"""
    A: np.ndarray
    source: np.ndarray
    target: np.ndarray
    rot: np.ndarray

    def apply(self, x):
        return self.target + (x - self.source) @ self.A.T


def _segment(src0, src1, dst0, dst1, parent_rot, stretch=True, limits=(0.5, 2.0)):
    """Affine mapping a source bone segment onto a destination segment."""
    dv, dh = src1 - src0, dst1 - dst0
    if np.linalg.norm(dv) < 1e-6 or np.linalg.norm(dh) < 1e-6:
        rot = parent_rot
        return Affine(rot.copy(), src0, dst0, rot)
    rot = rotation_between(parent_rot @ dv, dh) @ parent_rot
    A = rot.copy()
    if stretch:
        k = float(np.clip(np.linalg.norm(dh) / np.linalg.norm(dv), *limits))
        a = dh / np.linalg.norm(dh)
        A = (np.eye(3) + (k - 1.0) * np.outer(a, a)) @ rot
    return Affine(A, src0, dst0, rot)


@dataclass
class FitResult:
    positions: list      # per primitive (n,3)
    normals: list
    tangents: list
    functions: list      # per primitive (n,4) hunter bone function ids
    weights: list        # per primitive (n,4)
    scale: float
    notes: list


class HunterSkeleton:
    """Positions and parentage of the hunter bones, keyed by function id."""

    def __init__(self, positions, parents):
        self.pos = positions          # function -> (3,)
        self.parent = parents         # function -> parent function or None

    def has(self, f):
        return f in self.pos

    def p(self, f):
        return self.pos[f]


def _pick_upper_arm(sk, candidates, elbow):
    best, best_d = None, -1.0
    for c in candidates:
        if sk.has(c) and sk.has(elbow):
            d = np.linalg.norm(sk.p(c) - sk.p(elbow))
            if d > best_d:
                best, best_d = c, d
    return best


def fit_vrm(model, sk, height_scale=1.0):
    notes = []
    H = model.humanoid

    def vpos(name):
        return model.bone_position(name)

    required = ["hips", "head", "leftUpperArm", "rightUpperArm", "leftUpperLeg", "rightUpperLeg",
                "leftFoot", "rightFoot"]
    missing = [r for r in required if r not in H]
    if missing:
        raise ValueError("VRM is missing required humanoid bones: " + ", ".join(missing))
    for f in (HEAD, LEFT["thigh"], RIGHT["thigh"], LEFT["foot"], RIGHT["foot"]):
        if not sk.has(f):
            raise ValueError("hunter skeleton is missing bone function %d" % f)

    up_l = _pick_upper_arm(sk, LEFT["upper"], LEFT["fore"]) or LEFT["shoulder"]
    up_r = _pick_upper_arm(sk, RIGHT["upper"], RIGHT["fore"]) or RIGHT["shoulder"]
    if not (sk.has(up_l) and sk.has(up_r)):
        raise ValueError("hunter skeleton has no upper arm bones")

    # --- 1. global orientation ---------------------------------------------
    v_hipmid = (vpos("leftUpperLeg") + vpos("rightUpperLeg")) / 2
    v_neck = vpos("neck") if "neck" in H else vpos("head")
    v_up = v_neck - v_hipmid
    v_left = vpos("leftUpperArm") - vpos("rightUpperArm")
    h_hipmid = (sk.p(LEFT["thigh"]) + sk.p(RIGHT["thigh"])) / 2
    h_neck = sk.p(NECK) if sk.has(NECK) else sk.p(HEAD)
    h_up = h_neck - h_hipmid
    h_left = sk.p(up_l) - sk.p(up_r)
    # glTF is right-handed, so a VRM's forward is left x up.
    v_fwd = np.cross(v_left, v_up)
    # The hunter's forward comes from where its toes point; the game's axes may be
    # mirrored relative to glTF, in which case the fit includes a reflection.
    h_fwd = None
    if sk.has(LEFT["toe"]) and sk.has(RIGHT["toe"]):
        h_fwd = (sk.p(LEFT["toe"]) - sk.p(LEFT["foot"]) + sk.p(RIGHT["toe"]) - sk.p(RIGHT["foot"])) / 2
        h_fwd = h_fwd - h_up * np.dot(h_fwd, h_up) / max(np.dot(h_up, h_up), 1e-12)
        if np.linalg.norm(h_fwd) < 1e-3 * np.linalg.norm(h_up):
            h_fwd = None
    if h_fwd is None:
        h_fwd = np.cross(h_left, h_up)
        notes.append("hunter toes not found; assuming right-handed axes")
    fv = frame(v_up, v_fwd)
    fh = frame(h_up, h_fwd)
    if fv is None or fh is None:
        raise ValueError("could not work out which way the model is facing")
    # Third axis: the side that is "left" in each skeleton (not a cross product).
    fv[:, 2] = _unit(v_left - fv[:, 0] * np.dot(fv[:, 0], v_left) - fv[:, 1] * np.dot(fv[:, 1], v_left))
    fh[:, 2] = _unit(h_left - fh[:, 0] * np.dot(fh[:, 0], h_left) - fh[:, 1] * np.dot(fh[:, 1], h_left))
    R = fh @ fv.T
    mirror = np.linalg.det(R) < 0
    if mirror:
        notes.append("game axes are mirrored relative to glTF; model reflected to match")

    # --- 2. uniform scale from ankle-to-head height --------------------------
    up_h = fh[:, 0]
    up_v = fv[:, 0]
    v_feet = (vpos("leftFoot") + vpos("rightFoot")) / 2
    h_feet = (sk.p(LEFT["foot"]) + sk.p(RIGHT["foot"])) / 2
    v_height = float(np.dot(vpos("head") - v_feet, up_v))
    h_height = float(np.dot(sk.p(HEAD) - h_feet, up_h))
    if v_height <= 0 or h_height <= 0:
        raise ValueError("could not measure model height")
    s = h_height / v_height * height_scale
    notes.append("scale %.4f (model %.3f -> hunter %.3f units)" % (s, v_height, h_height))

    def to_hunter(x):
        return (x - v_feet) @ (s * R).T + h_feet

    nodes_w = np.array([to_hunter(model.world[i][:3, 3]) for i in range(len(model.nodes))])

    def V(name):
        return nodes_w[H[name]] if name in H else None

    I = np.eye(3)
    T = {}  # canonical bone name -> Affine, in the already globally-aligned space

    # --- 3a. torso as one block ---------------------------------------------
    v_hipmid_w = (V("leftUpperLeg") + V("rightUpperLeg")) / 2
    v_neck_w = V("neck") if "neck" in H else V("head")
    t_hips = h_hipmid + (V("hips") - v_hipmid_w)
    torso = _segment(V("hips"), v_neck_w, t_hips, h_neck, I)
    for name in TORSO:
        if name in H:
            T[name] = torso
    q_torso = torso.rot

    # --- 3b. neck and head ---------------------------------------------------
    if "neck" in H and sk.has(NECK):
        T["neck"] = _segment(V("neck"), V("head"), sk.p(NECK), sk.p(HEAD), q_torso)
    T["head"] = Affine(q_torso.copy(), V("head"), sk.p(HEAD), q_torso)

    # --- 3c. arms, hands, fingers --------------------------------------------
    for side, ids, up_id in (("left", LEFT, up_l), ("right", RIGHT, up_r)):
        n = lambda b: side + b
        parent_rot = q_torso
        if n("Shoulder") in H and sk.has(ids["shoulder"]):
            seg = _segment(V(n("Shoulder")), V(n("UpperArm")), sk.p(ids["shoulder"]), sk.p(up_id), parent_rot)
            T[n("Shoulder")] = seg
        chain = [(n("UpperArm"), up_id), (n("LowerArm"), ids["fore"]), (n("Hand"), ids["hand"])]
        rot = parent_rot
        for (b0, f0), (b1, f1) in zip(chain, chain[1:]):
            if b0 in H and b1 in H and sk.has(f0) and sk.has(f1):
                seg = _segment(V(b0), V(b1), sk.p(f0), sk.p(f1), rot)
                T[b0] = seg
                rot = seg.rot
        # Hand: match the whole hand frame (finger direction + knuckle line).
        hand = n("Hand")
        if hand in H and sk.has(ids["hand"]):
            q_hand = rot
            mid_v = V(n("MiddleProximal"))
            idx_v, lit_v = V(n("IndexProximal")), V(n("LittleProximal"))
            mid_h = ids["middle"][0]
            if mid_v is not None and idx_v is not None and lit_v is not None and \
                    sk.has(mid_h) and sk.has(ids["index"][0]) and sk.has(ids["little"][0]):
                fv_h = frame(mid_v - V(hand), idx_v - lit_v)
                fh_h = frame(sk.p(mid_h) - sk.p(ids["hand"]), sk.p(ids["index"][0]) - sk.p(ids["little"][0]))
                if fv_h is not None and fh_h is not None:
                    q_hand = fh_h @ fv_h.T
            T[hand] = Affine(q_hand.copy(), V(hand), sk.p(ids["hand"]), q_hand)
            for vname, key, parts in FINGERS:
                bones = [n(vname + p) for p in parts]
                funcs = ids[key]
                frot = q_hand
                for i, (b, f) in enumerate(zip(bones, funcs)):
                    if b not in H or not sk.has(f):
                        break
                    if i + 1 < len(bones) and bones[i + 1] in H and sk.has(funcs[i + 1]):
                        seg = _segment(V(b), V(bones[i + 1]), sk.p(f), sk.p(funcs[i + 1]), frot)
                    else:
                        seg = Affine(frot.copy(), V(b), sk.p(f), frot)
                    T[b] = seg
                    frot = seg.rot

    # --- 3d. legs and feet ----------------------------------------------------
    for side, ids in (("left", LEFT), ("right", RIGHT)):
        n = lambda b: side + b
        rot = q_torso
        chain = [(n("UpperLeg"), ids["thigh"]), (n("LowerLeg"), ids["calf"]), (n("Foot"), ids["foot"])]
        for (b0, f0), (b1, f1) in zip(chain, chain[1:]):
            if b0 in H and b1 in H and sk.has(f0) and sk.has(f1):
                seg = _segment(V(b0), V(b1), sk.p(f0), sk.p(f1), rot)
                T[b0] = seg
                rot = seg.rot
        foot = n("Foot")
        if foot in H:
            if n("Toes") in H and sk.has(ids["toe"]):
                seg = _segment(V(foot), V(n("Toes")), sk.p(ids["foot"]), sk.p(ids["toe"]), q_torso, stretch=False)
                T[foot] = seg
                T[n("Toes")] = Affine(seg.rot.copy(), V(n("Toes")), sk.p(ids["toe"]), seg.rot)
            else:
                T[foot] = Affine(q_torso.copy(), V(foot), sk.p(ids["foot"]), q_torso)

    # --- 4. weights: VRM node -> canonical humanoid bone -> hunter function ----
    node_to_bone = {node: name for name, node in H.items()}
    weight_target = {
        "hips": PELVIS, "spine": WAIST, "chest": CHEST, "upperChest": CHEST,
        "neck": NECK, "head": HEAD, "leftEye": HEAD, "rightEye": HEAD, "jaw": HEAD,
    }
    for side, ids, up_id in (("left", LEFT, up_l), ("right", RIGHT, up_r)):
        weight_target[side + "Shoulder"] = ids["shoulder"]
        weight_target[side + "UpperArm"] = up_id
        weight_target[side + "LowerArm"] = ids["fore"]
        weight_target[side + "Hand"] = ids["hand"]
        weight_target[side + "UpperLeg"] = ids["thigh"]
        weight_target[side + "LowerLeg"] = ids["calf"]
        weight_target[side + "Foot"] = ids["foot"]
        weight_target[side + "Toes"] = ids["toe"]
        for vname, key, parts in FINGERS:
            for p, f in zip(parts, ids[key]):
                weight_target[side + vname + p] = f

    def humanoid_ancestor(node):
        while node >= 0:
            if node in node_to_bone and node_to_bone[node] in weight_target:
                return node_to_bone[node]
            node = model.parents[node]
        return "hips"

    def resolve_function(bone):
        """Hunter function for a canonical bone, walking up if the skeleton lacks it."""
        cur = bone
        while cur is not None:
            f = weight_target.get(cur)
            if f is not None and sk.has(f):
                return f
            cur = _humanoid_parent(cur)
        for f in (PELVIS, WAIST, ROOT):
            if sk.has(f):
                return f
        return next(iter(sk.pos))

    def transform_for(bone):
        cur = bone
        while cur is not None:
            if cur in T:
                return T[cur]
            cur = _humanoid_parent(cur)
        return torso

    node_bone = {}
    for node in range(len(model.nodes)):
        node_bone[node] = humanoid_ancestor(node)

    twist = {}
    for ids in (LEFT, RIGHT):
        tw = ids["fore_twist"]
        if sk.has(tw) and sk.parent.get(tw) == ids["fore"] and sk.has(ids["hand"]):
            twist[ids["fore"]] = (tw, sk.p(ids["fore"]), sk.p(ids["hand"]))

    out_pos, out_nrm, out_tan, out_fn, out_w = [], [], [], [], []
    for prim in model.primitives:
        pos = to_hunter(prim.positions)
        nrm = prim.normals @ R.T
        hand_sign = -1.0 if mirror else 1.0
        tan = None if prim.tangents is None else \
            np.hstack([prim.tangents[:, :3] @ R.T, prim.tangents[:, 3:4] * hand_sign])
        n = len(pos)
        newp = np.zeros_like(pos)
        newn = np.zeros_like(nrm)
        newt = np.zeros((n, 3)) if tan is not None else None
        funcs = np.zeros((n, 4), dtype=np.int64)
        uniq = np.unique(prim.joints)
        for node in uniq:
            bone = node_bone[int(node)]
            aff = transform_for(bone)
            func = resolve_function(bone)
            mask = prim.joints == node
            wk = np.where(mask, prim.weights, 0.0).sum(axis=1)
            sel = wk > 0
            if np.any(sel):
                newp[sel] += wk[sel, None] * aff.apply(pos[sel])
                ninv = np.linalg.inv(aff.A).T
                newn[sel] += wk[sel, None] * (nrm[sel] @ ninv.T)
                if newt is not None:
                    newt[sel] += wk[sel, None] * (tan[sel, :3] @ aff.A.T)
            funcs[mask] = func
        ln = np.linalg.norm(newn, axis=1, keepdims=True)
        ln[ln < 1e-12] = 1
        newn /= ln
        if newt is not None:
            newt -= newn * np.sum(newn * newt, axis=1, keepdims=True)
            lt = np.linalg.norm(newt, axis=1, keepdims=True)
            lt[lt < 1e-12] = 1
            newt = np.hstack([newt / lt, tan[:, 3:4]])

        f_ids, f_w = _merge_weights(funcs, prim.weights)
        if twist:
            f_ids, f_w = _apply_twist(newp, f_ids, f_w, twist)
        out_pos.append(newp)
        out_nrm.append(newn)
        out_tan.append(newt)
        out_fn.append(f_ids)
        out_w.append(f_w)
    return FitResult(out_pos, out_nrm, out_tan, out_fn, out_w, s, notes)


_PARENT = {
    "spine": "hips", "chest": "spine", "upperChest": "chest", "neck": "upperChest", "head": "neck",
    "leftEye": "head", "rightEye": "head", "jaw": "head",
}
for _s in ("left", "right"):
    _PARENT.update({
        _s + "Shoulder": "upperChest", _s + "UpperArm": _s + "Shoulder", _s + "LowerArm": _s + "UpperArm",
        _s + "Hand": _s + "LowerArm", _s + "UpperLeg": "hips", _s + "LowerLeg": _s + "UpperLeg",
        _s + "Foot": _s + "LowerLeg", _s + "Toes": _s + "Foot",
    })
    for _v, _k, _parts in FINGERS:
        prev = _s + "Hand"
        for _p in _parts:
            _PARENT[_s + _v + _p] = prev
            prev = _s + _v + _p


def _humanoid_parent(bone):
    return _PARENT.get(bone)


def _merge_weights(funcs, weights, max_influences=4):
    """Sum weights per function id, keep the strongest four, renormalise."""
    n = len(funcs)
    out_ids = np.zeros((n, max_influences), dtype=np.int64)
    out_w = np.zeros((n, max_influences))
    for i in range(n):
        acc = {}
        for f, w in zip(funcs[i], weights[i]):
            if w > 0:
                acc[int(f)] = acc.get(int(f), 0.0) + float(w)
        if not acc:
            acc = {int(funcs[i][0]): 1.0}
        items = sorted(acc.items(), key=lambda kv: -kv[1])[:max_influences]
        total = sum(w for _, w in items)
        for k, (f, w) in enumerate(items):
            out_ids[i, k] = f
            out_w[i, k] = w / total
        for k in range(len(items), max_influences):
            out_ids[i, k] = items[0][0]
    return out_ids, out_w


def _apply_twist(pos, ids, w, twist):
    """Hand some forearm weight near the wrist to the forearm twist bone."""
    ids, w = ids.copy(), w.copy()
    for fore, (tw, p0, p1) in twist.items():
        axis = p1 - p0
        length2 = float(np.dot(axis, axis))
        if length2 < 1e-12:
            continue
        rows, cols = np.nonzero((ids == fore) & (w > 0))
        for r, c in zip(rows, cols):
            t = float(np.dot(pos[r] - p0, axis) / length2)
            share = float(np.clip((t - 0.2) / 0.8, 0.0, 1.0)) * 0.7
            if share <= 0:
                continue
            moved = w[r, c] * share
            w[r, c] -= moved
            acc = {}
            for f, x in zip(ids[r], w[r]):
                if x > 0:
                    acc[int(f)] = acc.get(int(f), 0.0) + float(x)
            acc[tw] = acc.get(tw, 0.0) + moved
            items = sorted(acc.items(), key=lambda kv: -kv[1])[:4]
            total = sum(x for _, x in items)
            ids[r] = items[0][0]
            w[r] = 0
            for k, (f, x) in enumerate(items):
                ids[r, k] = f
                w[r, k] = x / total
    return ids, w
