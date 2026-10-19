"""Synthetic test data: a small humanoid VRM and a fake hunter armor set.

The fake hunter deliberately differs from the VRM: Z-up instead of Y-up,
centimetre-ish units, a different facing direction and an A-pose, so the
fitting code has real work to do.
"""

import io
import json
import math
import os
import struct

import numpy as np

from vroidhunter import mod3 as M
from vroidhunter import mrl3 as R
from vroidhunter.binio import jamcrc

# --------------------------------------------------------------------------
# VRM
# --------------------------------------------------------------------------

# name, parent, position (metres, Y up, model facing +Z, left = +X)
VRM_BONES = [
    ("hips", None, (0, 0.95, 0)),
    ("spine", "hips", (0, 1.05, 0)),
    ("chest", "spine", (0, 1.18, 0)),
    ("neck", "chest", (0, 1.40, 0)),
    ("head", "neck", (0, 1.48, 0)),
    ("leftShoulder", "chest", (0.03, 1.35, 0)),
    ("leftUpperArm", "leftShoulder", (0.15, 1.35, 0)),
    ("leftLowerArm", "leftUpperArm", (0.40, 1.35, 0)),
    ("leftHand", "leftLowerArm", (0.62, 1.35, 0)),
    ("leftIndexProximal", "leftHand", (0.70, 1.35, 0.02)),
    ("leftMiddleProximal", "leftHand", (0.71, 1.35, 0.0)),
    ("leftLittleProximal", "leftHand", (0.69, 1.35, -0.025)),
    ("rightShoulder", "chest", (-0.03, 1.35, 0)),
    ("rightUpperArm", "rightShoulder", (-0.15, 1.35, 0)),
    ("rightLowerArm", "rightUpperArm", (-0.40, 1.35, 0)),
    ("rightHand", "rightLowerArm", (-0.62, 1.35, 0)),
    ("rightIndexProximal", "rightHand", (-0.70, 1.35, 0.02)),
    ("rightMiddleProximal", "rightHand", (-0.71, 1.35, 0.0)),
    ("rightLittleProximal", "rightHand", (-0.69, 1.35, -0.025)),
    ("leftUpperLeg", "hips", (0.09, 0.90, 0)),
    ("leftLowerLeg", "leftUpperLeg", (0.09, 0.50, 0)),
    ("leftFoot", "leftLowerLeg", (0.09, 0.08, 0)),
    ("leftToes", "leftFoot", (0.09, 0.02, 0.12)),
    ("rightUpperLeg", "hips", (-0.09, 0.90, 0)),
    ("rightLowerLeg", "rightUpperLeg", (-0.09, 0.50, 0)),
    ("rightFoot", "rightLowerLeg", (-0.09, 0.08, 0)),
    ("rightToes", "rightFoot", (-0.09, 0.02, 0.12)),
]
SEGMENTS = [("hips", "spine"), ("spine", "chest"), ("chest", "neck"), ("neck", "head"),
            ("leftUpperArm", "leftLowerArm"), ("leftLowerArm", "leftHand"), ("leftHand", "leftMiddleProximal"),
            ("rightUpperArm", "rightLowerArm"), ("rightLowerArm", "rightHand"), ("rightHand", "rightMiddleProximal"),
            ("leftUpperLeg", "leftLowerLeg"), ("leftLowerLeg", "leftFoot"), ("leftFoot", "leftToes"),
            ("rightUpperLeg", "rightLowerLeg"), ("rightLowerLeg", "rightFoot"), ("rightFoot", "rightToes")]


def _png(width, height, fill, transparent_right_half=False):
    from PIL import Image
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[:] = fill
    if transparent_right_half:
        arr[:, width // 2:, 3] = 0
    buf = io.BytesIO()
    Image.fromarray(arr, "RGBA").save(buf, "PNG")
    return buf.getvalue()


def _tube(p0, p1, radius, sides=6, rings=4):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    axis = p1 - p0
    a = axis / np.linalg.norm(axis)
    ref = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(a, ref)
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    pos, nrm, ts = [], [], []
    for r in range(rings):
        t = r / (rings - 1)
        for s in range(sides):
            ang = 2 * math.pi * s / sides
            d = math.cos(ang) * u + math.sin(ang) * v
            pos.append(p0 + axis * t + d * radius)
            nrm.append(d)
            ts.append(t)
    faces = []
    for r in range(rings - 1):
        for s in range(sides):
            a0 = r * sides + s
            a1 = r * sides + (s + 1) % sides
            b0 = a0 + sides
            b1 = a1 + sides
            faces += [(a0, a1, b1), (a0, b1, b0)]
    # Orient faces so their winding matches the outward normal (CCW = front, glTF).
    P = np.array(pos)
    F = np.array(faces)
    fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    vn = np.array(nrm)[F].sum(axis=1)
    flip = np.sum(fn * vn, axis=1) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    return P, np.array(nrm), np.array(ts), F


def _proportioned(name, p, leg_scale, arm_scale):
    x, y, z = p
    if ("Leg" in name or "Foot" in name or "Toes" in name) and y < 0.95:
        y = 0.95 - (0.95 - y) * leg_scale
    if any(k in name for k in ("LowerArm", "Hand", "Proximal")) and abs(x) > 0.15:
        x = np.sign(x) * (0.15 + (abs(x) - 0.15) * arm_scale)
    return (x, y, z)


def make_vrm(version=0, facing=+1, leg_scale=1.0, arm_scale=1.0):
    """Build a .vrm (glb) in memory. facing=+1: model looks along +Z, -1: along -Z.
    leg_scale / arm_scale stretch the avatar's limbs to give it non-hunter proportions."""
    names = [b[0] for b in VRM_BONES]
    index = {n: i for i, n in enumerate(names)}
    flip = np.array([facing, 1, facing], float)   # rotate 180 deg about Y when facing -Z
    world = {b[0]: np.array(_proportioned(b[0], b[2], leg_scale, arm_scale), float) * flip for b in VRM_BONES}

    nodes = []
    for name, parent, pos in VRM_BONES:
        local = world[name] - (world[parent] if parent else 0)
        nodes.append({"name": name, "translation": list(map(float, local)), "children": []})
    for i, (name, parent, _) in enumerate(VRM_BONES):
        if parent:
            nodes[index[parent]]["children"].append(i)
    # A hair bone that is not humanoid.
    hair_index = len(nodes)
    nodes.append({"name": "J_Sec_Hair1", "translation": [0.0, 0.12, -0.05 * facing], "children": []})
    nodes[index["head"]]["children"].append(hair_index)
    hair2_index = len(nodes)
    nodes.append({"name": "J_Sec_Hair2", "translation": [0.0, -0.06, 0.0], "children": []})
    nodes[hair_index]["children"].append(hair2_index)
    joint_nodes = list(range(len(nodes)))
    joint_world = {i: None for i in joint_nodes}
    for i in joint_nodes:
        joint_world[i] = world[names[i]] if i < len(names) else world["head"] + np.array([0, 0.12, -0.05 * facing])
    joint_world[hair2_index] = joint_world[hair_index] + np.array([0, -0.06, 0])

    # Body mesh: tubes along each segment, skinned to the first bone (blended at the joint).
    pos, nrm, uv, joints, weights, faces = [], [], [], [], [], []
    for b0, b1 in SEGMENTS:
        P, N, T, F = _tube(world[b0], world[b1], 0.04 if "Leg" in b0 or b0 in ("hips", "spine", "chest") else 0.025)
        base = len(pos)
        for p, n, t in zip(P, N, T):
            pos.append(p)
            nrm.append(n)
            uv.append((0.25, 0.5))
            j1 = index[b1] if b1 in index else index[b0]
            w1 = max(0.0, (t - 0.8) / 0.2) * 0.5
            joints.append((index[b0], j1, 0, 0))
            weights.append((1 - w1, w1, 0, 0))
        faces += [tuple(f + base) for f in F]
    body = dict(pos=np.array(pos, np.float32), nrm=np.array(nrm, np.float32), uv=np.array(uv, np.float32),
                joints=np.array(joints, np.uint16), weights=np.array(weights, np.float32),
                faces=np.array(faces, np.uint32))

    # Hair card: a quad, half of it transparent in its texture.
    h0 = joint_world[hair_index]
    hp = np.array([h0 + [-0.1, 0, 0], h0 + [0.1, 0, 0], h0 + [0.1, 0.1, 0], h0 + [-0.1, 0.1, 0],
                   h0 + [0.12, 0, 0], h0 + [0.3, 0, 0], h0 + [0.3, 0.1, 0], h0 + [0.12, 0.1, 0]], np.float32)
    hair = dict(pos=hp, nrm=np.tile(np.array([0, 0, facing], np.float32), (8, 1)),
                uv=np.array([[0.05, 1], [0.45, 1], [0.45, 0], [0.05, 0],
                             [0.55, 1], [0.95, 1], [0.95, 0], [0.55, 0]], np.float32),
                # bottom edge (y = h0) follows the second spring joint, top edge the first
                joints=np.array([[hair2_index if v in (0, 1, 4, 5) else hair_index, 0, 0, 0] for v in range(8)],
                                np.uint16),
                weights=np.tile(np.array([1, 0, 0, 0], np.float32), (8, 1)),
                # second quad sits on the transparent half (u > 0.5) and must be culled
                faces=np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], np.uint32))
    if facing < 0:
        hair["faces"] = hair["faces"][:, [0, 2, 1]]

    blob = bytearray()
    views, accessors = [], []

    def add(arr, target=None, kind=None):
        arr = np.ascontiguousarray(arr)
        while len(blob) % 4:
            blob.append(0)
        off = len(blob)
        blob.extend(arr.tobytes())
        views.append({"buffer": 0, "byteOffset": off, "byteLength": arr.nbytes})
        comp = {np.float32: 5126, np.uint16: 5123, np.uint32: 5125, np.uint8: 5121}[arr.dtype.type]
        typ = kind or {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4", 16: "MAT4"}[arr.shape[1] if arr.ndim > 1 else 1]
        acc = {"bufferView": len(views) - 1, "componentType": comp, "count": int(arr.shape[0]), "type": typ}
        if typ == "VEC3" and comp == 5126:
            acc["min"] = arr.min(axis=0).tolist()
            acc["max"] = arr.max(axis=0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    def prim(d, material):
        return {"attributes": {"POSITION": add(d["pos"]), "NORMAL": add(d["nrm"]), "TEXCOORD_0": add(d["uv"]),
                               "JOINTS_0": add(d["joints"]), "WEIGHTS_0": add(d["weights"])},
                "indices": add(d["faces"].reshape(-1)), "material": material}

    ibm = np.array([np.linalg.inv(_translation(joint_world[i])).T.reshape(16) for i in joint_nodes], np.float32)
    ibm_acc = add(ibm, kind="MAT4")
    mesh = {"primitives": [prim(body, 0), prim(hair, 1)]}
    img0 = _png(16, 16, (200, 150, 120, 255))
    img1 = _png(32, 16, (40, 30, 20, 255), transparent_right_half=True)
    images = []
    for data in (img0, img1):
        while len(blob) % 4:
            blob.append(0)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)})
        blob.extend(data)
        images.append({"bufferView": len(views) - 1, "mimeType": "image/png"})

    mesh_node = len(nodes)
    nodes.append({"name": "Body", "mesh": 0, "skin": 0})
    root_children = [index["hips"], mesh_node]
    js = {
        "asset": {"version": "2.0", "generator": "vroidhunter tests"},
        "scene": 0, "scenes": [{"nodes": root_children}],
        "nodes": nodes, "meshes": [mesh],
        "skins": [{"joints": joint_nodes, "inverseBindMatrices": ibm_acc}],
        "materials": [
            {"name": "Body_SKIN", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}},
            {"name": "Hair 01", "pbrMetallicRoughness": {"baseColorTexture": {"index": 1}},
             "alphaMode": "MASK", "alphaCutoff": 0.5, "doubleSided": True},
        ],
        "textures": [{"source": 0}, {"source": 1}], "images": images,
        "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(blob)}],
    }
    for n in js["nodes"]:
        if not n.get("children"):
            n.pop("children", None)
    if version == 0:
        js["extensions"] = {"VRM": {"meta": {"title": "Test Avatar"},
                                    "humanoid": {"humanBones": [{"bone": n, "node": index[n]} for n in names]},
                                    "secondaryAnimation": {"boneGroups": [
                                        {"stiffiness": 1.0, "gravityPower": 0, "dragForce": 0.4,
                                         "hitRadius": 0.02, "bones": [hair_index], "colliderGroups": [0]}],
                                        "colliderGroups": [{"node": index["head"], "colliders": [
                                            {"offset": {"x": 0, "y": 0.08, "z": 0}, "radius": 0.09}]}]}}}
    else:
        js["extensions"] = {"VRMC_vrm": {"specVersion": "1.0", "meta": {"name": "Test Avatar"},
                                         "humanoid": {"humanBones": {n: {"node": index[n]} for n in names}}},
                             "VRMC_springBone": {"specVersion": "1.0", "springs": [
                                 {"name": "hair", "joints": [{"node": hair_index, "stiffness": 1.0},
                                                             {"node": hair2_index, "stiffness": 1.0}]}],
                                 "colliders": [{"node": index["head"], "shape": {"sphere": {
                                     "offset": [0, 0.08, 0], "radius": 0.09}}}]}}
    js_bytes = json.dumps(js).encode()
    while len(js_bytes) % 4:
        js_bytes += b" "
    while len(blob) % 4:
        blob.append(0)
    total = 12 + 8 + len(js_bytes) + 8 + len(blob)
    out = struct.pack("<4sII", b"glTF", 2, total)
    out += struct.pack("<II", len(js_bytes), 0x4E4F534A) + js_bytes
    out += struct.pack("<II", len(blob), 0x004E4942) + bytes(blob)
    return out, {n: world[n] for n in names}


def _translation(p):
    m = np.eye(4)
    m[:3, 3] = p
    return m


# --------------------------------------------------------------------------
# Fake hunter armor
# --------------------------------------------------------------------------

def hunter_bones(scale=1.0):
    """function -> (parent function, world position). Z up, hunter faces -Y, left = -X."""
    c, s = math.cos(math.radians(40)), math.sin(math.radians(40))
    B = {0: (None, (0, 0, 0)), 1: (0, (0, 0, 100)), 13: (1, (0, 0, 96)), 2: (1, (0, 0, 122)),
         3: (2, (0, 0, 147)), 4: (3, (0, 0, 156))}
    for side, sx, ids in (("L", -1, (5, 6, 80, 7, 81, 8, 34, 37, 44, 31, 14, 15, 16, 17)),
                          ("R", 1, (9, 10, 82, 11, 83, 12, 51, 54, 61, 48, 18, 19, 20, 21))):
        sh, up, uptw, fo, fotw, ha, ix, mi, li, th, tg, ca, ft, to = ids
        B[sh] = (2, (sx * 4, 0, 142))
        B[up] = (sh, (sx * 17, 0, 140))
        B[uptw] = (up, (sx * (17 + 13 * c), 0, 140 - 13 * s))
        elbow = (sx * (17 + 27 * c), 0, 140 - 27 * s)
        B[fo] = (up, elbow)
        B[fotw] = (fo, (sx * (17 + 39 * c), 0, 140 - 39 * s))
        wrist = (sx * (17 + 51 * c), 0, 140 - 51 * s)
        B[ha] = (fo, wrist)
        d = np.array([sx * c, 0, -s])
        w = np.array(wrist, float)
        # the hunter faces -Y, so the thumb/index side is towards -Y
        B[mi] = (ha, tuple(w + d * 8))
        B[ix] = (ha, tuple(w + d * 7.5 + np.array([0, -2.2, 0])))
        B[li] = (ha, tuple(w + d * 7 + np.array([0, 2.5, 0])))
        B[th] = (ha, tuple(w + d * 3 + np.array([0, -3, 0])))
        B[tg] = (13, (sx * 10, 0, 93))
        B[ca] = (tg, (sx * 10, 0, 51))
        B[ft] = (ca, (sx * 10, 0, 9))
        B[to] = (ft, (sx * 10, -13, 2))
    return {f: (p, tuple(np.array(pos, float) * scale)) for f, (p, pos) in B.items()}


def _colmajor(m):
    return m.T.reshape(16).tolist()


def make_hunter_mod3(bones, material_names=("Ch_Pl_Body_Mt",), flip_winding=False, offset=(0, 0, 0),
                     box=((-15, 15), (-10, 10), (95, 145)), weight_func=2, extra_parts=()):
    funcs = sorted(bones)
    index = {f: i for i, f in enumerate(funcs)}
    world = {f: np.array(bones[f][1], float) + offset for f in funcs}
    skel = bytearray()
    for f in funcs:
        parent = bones[f][0]
        local = world[f] - (world[parent] if parent is not None else 0)
        skel += struct.pack(M.BONE_FMT, f, index[parent] if parent is not None else 255, 255, 0.0,
                            float(np.linalg.norm(local)), *local)
    for f in funcs:
        parent = bones[f][0]
        local = world[f] - (world[parent] if parent is not None else 0)
        skel += struct.pack("<16f", *_colmajor(_translation(local)))
    for f in funcs:
        skel += struct.pack("<16f", *_colmajor(np.linalg.inv(_translation(world[f]))))
    remap = bytearray([255] * 512)
    for f in funcs:
        remap[f] = index[f]
    skel += remap
    lm = np.array([_translation(world[f] - (world[bones[f][0]] if bones[f][0] is not None else 0)) for f in funcs])
    am = np.array([np.linalg.inv(_translation(world[f])) for f in funcs])
    header = {k: 0 for k in ("id", "version", "version2", "boneCount", "meshCount", "materialCount",
                             "vertexCount", "faceCount", "vertexIds", "vertexBufferSize", "secondBufferSize",
                             "groupCount", "creationDate", "boneOffset", "groupOffset", "materialNamesOffset",
                             "meshOffset", "vertexOffset", "facesOffset", "trailOffset", "unkOffset2")}
    header.update(id=M.MOD3_MAGIC, version=237, creationDate=0x5C7F1234)
    extra = bytearray(M.HEADER_EXTRA_SIZE)
    struct.pack_into("<f", extra, 48, 1000.0)
    struct.pack_into("<5f", extra, 48 + 76, 360, 60, 20, 10, 10)
    ref = M.Mod3(header, bytes(extra),
                 [M.Bone(f, index[bones[f][0]] if bones[f][0] is not None else 255, 255, 0.0, 0.0, tuple(world[f]))
                  for f in funcs], lm, am, bytes(remap), bytes(skel),
                 struct.pack("<i3i4f", 0, 0, 0, 0, 0, 0, 0, 0) * 2, list(material_names), [], b"\x01\x02\x03\x04")
    # A torso box so the reference has geometry for the winding check.
    (x0, x1), (y0, y1), (z0, z1) = box
    cube = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float) + offset
    center = cube.mean(axis=0)
    faces = []
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    for q in quads:
        faces += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
    F = np.array(faces)
    fn = np.cross(cube[F[:, 1]] - cube[F[:, 0]], cube[F[:, 2]] - cube[F[:, 0]])
    out = (cube[F].mean(axis=1) - center)
    wrong = np.sum(fn * out, axis=1) < 0
    F[wrong] = F[wrong][:, [0, 2, 1]]
    if flip_winding:
        F = F[:, [0, 2, 1]]
    nrm = (cube - center) / np.linalg.norm(cube - center, axis=1, keepdims=True)
    part = M.MeshPart(0, cube, nrm, np.tile([1.0, 0, 0, 1], (8, 1)), np.zeros((8, 2)),
                      np.full((8, 4), index[weight_func]), np.tile([1.0, 0, 0, 0], (8, 1)), F)
    tmpl = dict(shadowCast=0x13, visibleCondition=0, weightDynamics=25, unkn3=0, unknownIndex=0,
                mapData0=-1, mapData1=-1, intUnknown=0)
    extra = [ep(index, np.array(offset, float)) for ep in extra_parts]
    data = M.write_mod3(ref, [part] + extra, list(material_names), M.BLOCK_SKIN4_1UV, tmpl)
    return data


def make_hunter_mrl3(material_names=("Ch_Pl_Body_Mt",), tex_dir="pl\\f_equip\\pl001_0000\\body\\tex\\",
                     metal_first=False):
    """Player materials with the real PL constant buffer layout.

    metal_first: put a shiny metal material (film + metalness, no wetness) first, like a
    plate-armor set would have, to check that a plain material is preferred as template."""
    textures = [(0x241F5DEB, (0,) * 12, tex_dir + n) for n in
                ("f_body001_BML", "f_body001_NM", "f_body001_RMT", "f_body001_CMM", "common_special")]

    def mt(name, low=0x123):
        return ((jamcrc(name) & 0xFFFFF) << 12) | low

    def material(name, metal):
        binds = [R.Binding(0, (0, 0, 0), mt("CBMhMaterialPLLocal__disclosure"), 0, 0)]
        if not metal:
            binds.append(R.Binding(0, (0, 0, 0), mt("CBMhMaterialGlobal"), 0, 0))
        binds += [R.Binding(1, (0, 0, 0), mt("SSBase"), 0, 0),
                  R.Binding(2, (0, 0, 0), mt("tAlbedoMap__disclosure"), 1, 0),
                  R.Binding(2, (0, 0, 0), mt("tNormalMap__disclosure"), 2, 0),
                  R.Binding(2, (0, 0, 0), mt("tRMTMap__disclosure"), 3, 0),
                  R.Binding(2, (0, 0, 0), mt("tColorMaskMap__disclosure"), 4, 0),
                  R.Binding(2, (0, 0, 0), mt("tSomeUnknownMap__disclosure"), 5, 0)]
        m = R.Material([0x4516E7AB, jamcrc(name), 0xBEEF0000 + (1 if metal else 2), 0x1234, 0, 0, 0]
                       + [0] * 9 + [0] + [0] * 15 + [0, 0], binds, bytes(336 + (0 if metal else 32)))
        R.set_fields(m, {"fBaseMapFactor__uiColor": (1, 1, 1, 1), "bUseCMM": 1,
                         "fMetalic__uiUNorm": 0.9 if metal else 0.0, "fFilmBlend__uiUNorm": 0.5 if metal else 0.0,
                         "fMaterialWetBlend": 0.25})
        return m

    materials = []
    if metal_first:
        materials.append(material("Ch_Pl_Metal_Mt", True))
    for name in material_names:
        materials.append(material(name, False))
    head = [R.MRL3_MAGIC, 12, 0, 0, 0, 80, 96, 132, 93, 0, 0, 0, 0, 0, 0, 0, 0]
    return R.write_mrl3(R.Mrl3(head, textures, materials))


LEG_BONES = {14, 15, 16, 17, 18, 19, 20, 21}


def arm_tube(bones, fore, hand, twist):
    """Reference gauntlet: forearm weights move from the forearm bone to the twist bone
    along the forearm (twist share = t), like real MHW armor."""
    def make(index, offset):
        a, b = np.array(bones[fore][1]) + offset, np.array(bones[hand][1]) + offset
        P, N, T, F = _tube(a, b, 4.0, sides=8, rings=9)
        ids = np.zeros((len(P), 4), dtype=np.int64)
        w = np.zeros((len(P), 4))
        ids[:, 0], ids[:, 1] = index[fore], index[twist]
        w[:, 0], w[:, 1] = 1 - T, T
        return M.MeshPart(0, P, N, np.tile([1.0, 0, 0, 1], (len(P), 1)), np.zeros((len(P), 2)), ids, w, F)
    return make


def make_game_folder(root, armor="pl001_0000", genders=("f", "m"), flip_winding=False, partial_body=False):
    """Write a fake extracted-game folder that FolderFiles can read.

    partial_body: the body model's skeleton has no leg bones (only the leg piece does).
    """
    for g in genders:
        bones = hunter_bones(1.0 if g == "f" else 1.08)
        base = os.path.join(root, "pl", "%s_equip" % g, armor)
        for part in ("helm", "body", "arm", "wst", "leg"):
            num = armor[2:5]
            stem = "%s_%s%s" % (g, part, num)
            d = os.path.join(base, part, "mod")
            os.makedirs(d, exist_ok=True)
            mats = ("Ch_Pl_%s_Mt" % part.capitalize(),)
            piece_bones = bones
            if partial_body and part == "body":
                piece_bones = {k: v for k, v in bones.items() if k not in LEG_BONES}
            extra = ()
            if part == "arm":
                extra = (arm_tube(piece_bones, 7, 8, 81), arm_tube(piece_bones, 11, 12, 83))
            with open(os.path.join(d, stem + ".mod3"), "wb") as f:
                f.write(make_hunter_mod3(piece_bones, mats, flip_winding, extra_parts=extra))
            with open(os.path.join(d, stem + ".mrl3"), "wb") as f:
                f.write(make_hunter_mrl3(mats, "pl\\%s_equip\\%s\\%s\\tex\\" % (g, armor, part),
                                         metal_first=(part == "body")))
        face = os.path.join(root, "pl", "%s_face" % g, "face000", "mod")
        os.makedirs(face, exist_ok=True)
        with open(os.path.join(face, "face000.mod3"), "wb") as f:
            # The hunter's own neck: a box around the neck bone, thicker than the avatar's neck.
            f.write(make_hunter_mod3(bones, ("Ch_Pl_Face_Mt",), weight_func=3,
                                     box=((-5, 5), (-5, 5), (np.array(bones[3][1])[2] - 6, np.array(bones[3][1])[2] + 6))))
    hair = os.path.join(root, "pl", "hair", "hair000", "mod")
    os.makedirs(hair, exist_ok=True)
    with open(os.path.join(hair, "hair000.mod3"), "wb") as f:
        f.write(make_hunter_mod3(hunter_bones(), ("Ch_Pl_Hair_Mt",)))
