"""Loader for .vrm avatars (VRM 0.x and 1.0, both glTF 2.0 binaries).

Produces rest-pose geometry in world space together with per-vertex joint
influences expressed as glTF node indices, plus the humanoid bone map.
"""

import base64
import json
import os
import struct
from dataclasses import dataclass, field

import numpy as np

COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
TYPE_SIZE = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
NORMALIZE_DIV = {np.int8: 127.0, np.uint8: 255.0, np.int16: 32767.0, np.uint16: 65535.0}


@dataclass
class Material:
    name: str
    base_image: object = None       # image index or None
    color: tuple = (1.0, 1.0, 1.0, 1.0)
    alpha_mode: str = "OPAQUE"      # OPAQUE / MASK / BLEND
    alpha_cutoff: float = 0.5
    double_sided: bool = False
    normal_image: object = None


@dataclass
class Primitive:
    positions: np.ndarray
    normals: np.ndarray
    tangents: object
    uvs: np.ndarray
    joints: np.ndarray       # (n,4) glTF node indices
    weights: np.ndarray      # (n,4)
    faces: np.ndarray        # (m,3)
    material: int
    node: int


@dataclass
class SpringChain:
    """A linear chain of spring (physics) joints, root first."""
    nodes: list
    stiffness: float = 1.0
    gravity: float = 0.0
    drag: float = 0.4
    radius: float = 0.02


@dataclass
class Collider:
    """A spring-bone collider: sphere (tail None) or capsule, in the node's local space."""
    node: int
    offset: tuple
    radius: float
    tail: object = None


@dataclass
class VrmModel:
    version: int
    nodes: list
    parents: list
    world: np.ndarray        # (nodes,4,4)
    humanoid: dict           # canonical bone name -> node index
    primitives: list
    materials: list
    images: list             # raw image bytes per image index
    title: str = ""
    meta: dict = field(default_factory=dict)
    springs: list = field(default_factory=list)
    colliders: list = field(default_factory=list)

    def bone_position(self, name):
        node = self.humanoid.get(name)
        if node is None:
            return None
        return self.world[node][:3, 3].copy()


def _quat_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _node_matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    m = np.eye(4)
    s = np.array(node.get("scale", [1, 1, 1]), dtype=np.float64)
    r = _quat_to_matrix(node.get("rotation", [0, 0, 0, 1]))
    m[:3, :3] = r * s[None, :]
    m[:3, 3] = node.get("translation", [0, 0, 0])
    return m


def parse_glb(data):
    if data[:4] != b"glTF":
        # Plain JSON glTF is accepted too.
        return json.loads(data.decode("utf-8")), b""
    _, version, length = struct.unpack_from("<4sII", data, 0)
    if version != 2:
        raise ValueError("only glTF 2.0 is supported")
    pos = 12
    js, binary = None, b""
    while pos < min(length, len(data)):
        clen, ctype = struct.unpack_from("<II", data, pos)
        chunk = data[pos + 8:pos + 8 + clen]
        if ctype == 0x4E4F534A:
            js = json.loads(chunk.decode("utf-8"))
        elif ctype == 0x004E4942:
            binary = chunk
        pos += 8 + clen
    if js is None:
        raise ValueError("glb has no JSON chunk")
    return js, binary


class _Gltf:
    def __init__(self, js, binary, base_dir):
        self.js = js
        self.binary = binary
        self.base_dir = base_dir
        self._buffers = {}

    def buffer(self, index):
        if index in self._buffers:
            return self._buffers[index]
        buf = self.js["buffers"][index]
        uri = buf.get("uri")
        if uri is None:
            data = self.binary
        elif uri.startswith("data:"):
            data = base64.b64decode(uri.split(",", 1)[1])
        else:
            with open(os.path.join(self.base_dir, uri), "rb") as f:
                data = f.read()
        self._buffers[index] = data
        return data

    def view_bytes(self, index):
        view = self.js["bufferViews"][index]
        data = self.buffer(view["buffer"])
        start = view.get("byteOffset", 0)
        return data[start:start + view["byteLength"]], view.get("byteStride")

    def accessor(self, index):
        acc = self.js["accessors"][index]
        dtype = np.dtype(COMPONENT[acc["componentType"]]).newbyteorder("<")
        ncomp = TYPE_SIZE[acc["type"]]
        count = acc["count"]
        if "bufferView" in acc:
            raw, stride = self.view_bytes(acc["bufferView"])
            offset = acc.get("byteOffset", 0)
            elem = dtype.itemsize * ncomp
            if stride and stride != elem:
                arr = np.frombuffer(raw, dtype=np.uint8, count=stride * (count - 1) + elem, offset=offset)
                rows = np.lib.stride_tricks.as_strided(arr, shape=(count, elem), strides=(stride, 1))
                out = np.ascontiguousarray(rows).view(dtype).reshape(count, ncomp)
            else:
                out = np.frombuffer(raw, dtype=dtype, count=count * ncomp, offset=offset).reshape(count, ncomp)
            out = out.copy()
        else:
            out = np.zeros((count, ncomp), dtype=dtype)
        sparse = acc.get("sparse")
        if sparse:
            idx_info, val_info = sparse["indices"], sparse["values"]
            iraw, _ = self.view_bytes(idx_info["bufferView"])
            idx = np.frombuffer(iraw, dtype=np.dtype(COMPONENT[idx_info["componentType"]]).newbyteorder("<"),
                                count=sparse["count"], offset=idx_info.get("byteOffset", 0))
            vraw, _ = self.view_bytes(val_info["bufferView"])
            vals = np.frombuffer(vraw, dtype=dtype, count=sparse["count"] * ncomp,
                                 offset=val_info.get("byteOffset", 0)).reshape(-1, ncomp)
            out[idx.astype(np.int64)] = vals
        if acc.get("normalized") and dtype.type in NORMALIZE_DIV:
            out = np.maximum(out.astype(np.float64) / NORMALIZE_DIV[dtype.type], -1.0)
        return out

    def image_bytes(self, index):
        img = self.js["images"][index]
        if "bufferView" in img:
            return self.view_bytes(img["bufferView"])[0]
        uri = img.get("uri", "")
        if uri.startswith("data:"):
            return base64.b64decode(uri.split(",", 1)[1])
        path = os.path.join(self.base_dir, uri)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
        return None


def _canonical_humanoid(js):
    ext = js.get("extensions", {})
    raw = {}
    version = 0
    if "VRMC_vrm" in ext:
        version = 1
        for name, info in ext["VRMC_vrm"].get("humanoid", {}).get("humanBones", {}).items():
            if "node" in info:
                raw[name] = info["node"]
    elif "VRM" in ext:
        for item in ext["VRM"].get("humanoid", {}).get("humanBones", []):
            if "node" in item and item.get("node", -1) >= 0:
                raw[item["bone"]] = item["node"]
    else:
        raise ValueError("this file has no VRM humanoid data (is it a .vrm?)")
    out = {}
    for name, node in raw.items():
        canon = name
        for side in ("left", "right"):
            if version == 1:
                rename = {side + "ThumbMetacarpal": side + "Thumb1", side + "ThumbProximal": side + "Thumb2",
                          side + "ThumbDistal": side + "Thumb3"}
            else:
                rename = {side + "ThumbProximal": side + "Thumb1", side + "ThumbIntermediate": side + "Thumb2",
                          side + "ThumbDistal": side + "Thumb3"}
            canon = rename.get(canon, canon)
        out[canon] = node
    return version, out


def _materials(g):
    js = g.js
    vrm0_props = {}
    ext = js.get("extensions", {})
    if "VRM" in ext:
        for prop in ext["VRM"].get("materialProperties", []):
            vrm0_props[prop.get("name")] = prop

    def tex_image(tex_index):
        if tex_index is None:
            return None
        try:
            tex = js["textures"][tex_index]
        except (KeyError, IndexError):
            return None
        if "source" in tex:
            return tex["source"]
        for e in tex.get("extensions", {}).values():
            if "source" in e:
                return e["source"]
        return None

    mats = []
    for i, m in enumerate(js.get("materials", [])):
        name = m.get("name", "material%d" % i)
        pbr = m.get("pbrMetallicRoughness", {})
        base = tex_image(pbr.get("baseColorTexture", {}).get("index"))
        color = tuple(pbr.get("baseColorFactor", [1, 1, 1, 1]))
        alpha_mode = m.get("alphaMode", "OPAQUE")
        cutoff = m.get("alphaCutoff", 0.5)
        normal = tex_image(m.get("normalTexture", {}).get("index"))
        prop = vrm0_props.get(name)
        if prop:
            tp = prop.get("textureProperties", {})
            if base is None and "_MainTex" in tp:
                base = tex_image(tp["_MainTex"])
            if normal is None and "_BumpMap" in tp:
                normal = tex_image(tp["_BumpMap"])
            vp = prop.get("vectorProperties", {})
            if "_Color" in vp and "baseColorFactor" not in pbr:
                color = tuple(vp["_Color"])
            render = prop.get("tagMap", {}).get("RenderType", "")
            if render == "TransparentCutout":
                alpha_mode = "MASK"
            elif render == "Transparent":
                alpha_mode = "BLEND"
            cutoff = prop.get("floatProperties", {}).get("_Cutoff", cutoff)
        mats.append(Material(name=name, base_image=base, color=color, alpha_mode=alpha_mode,
                             alpha_cutoff=float(cutoff), double_sided=bool(m.get("doubleSided", False)),
                             normal_image=normal))
    return mats


def _children_map(nodes):
    return {i: list(n.get("children", [])) for i, n in enumerate(nodes)}


def _subtree_depth(children, node, memo):
    if node not in memo:
        memo[node] = 1 + max((_subtree_depth(children, c, memo) for c in children.get(node, [])), default=0)
    return memo[node]


def _linear_chains(children, root):
    """Split a joint tree into linear chains; the deepest branch continues each chain."""
    memo = {}
    chains, todo = [], [root]
    while todo:
        start = todo.pop(0)
        chain, cur = [start], start
        while children.get(cur):
            kids = sorted(children[cur], key=lambda c: -_subtree_depth(children, c, memo))
            todo.extend(kids[1:])
            cur = kids[0]
            chain.append(cur)
        chains.append(chain)
    return chains


def _springs(js):
    ext = js.get("extensions", {})
    nodes = js.get("nodes", [])
    out = []
    if "VRMC_springBone" in ext:
        for spring in ext["VRMC_springBone"].get("springs", []):
            joints = [j for j in spring.get("joints", []) if "node" in j]
            if len(joints) < 1:
                continue
            first = joints[0]
            out.append(SpringChain(
                nodes=[j["node"] for j in joints],
                stiffness=float(first.get("stiffness", 1.0)),
                gravity=float(first.get("gravityPower", 0.0)),
                drag=float(first.get("dragForce", 0.5)),
                radius=float(first.get("hitRadius", 0.02))))
    elif "VRM" in ext:
        children = _children_map(nodes)
        seen = set()
        for group in ext["VRM"].get("secondaryAnimation", {}).get("boneGroups", []):
            for root in group.get("bones", []):
                if root is None or root < 0 or root in seen:
                    continue
                for chain in _linear_chains(children, root):
                    chain = [n for n in chain if n not in seen]
                    seen.update(chain)
                    if chain:
                        out.append(SpringChain(
                            nodes=chain,
                            stiffness=float(group.get("stiffiness", group.get("stiffness", 1.0))),
                            gravity=float(group.get("gravityPower", 0.0)),
                            drag=float(group.get("dragForce", 0.4)),
                            radius=float(group.get("hitRadius", 0.02))))
    return out


def _colliders(js):
    ext = js.get("extensions", {})
    out = []
    if "VRMC_springBone" in ext:
        for c in ext["VRMC_springBone"].get("colliders", []):
            shape = c.get("shape", {})
            if "sphere" in shape:
                sp = shape["sphere"]
                out.append(Collider(c.get("node", -1), tuple(sp.get("offset", (0, 0, 0))), float(sp.get("radius", 0))))
            elif "capsule" in shape:
                cp = shape["capsule"]
                out.append(Collider(c.get("node", -1), tuple(cp.get("offset", (0, 0, 0))), float(cp.get("radius", 0)),
                                    tuple(cp.get("tail", (0, 0, 0)))))
    elif "VRM" in ext:
        for group in ext["VRM"].get("secondaryAnimation", {}).get("colliderGroups", []):
            node = group.get("node", -1)
            for c in group.get("colliders", []):
                o = c.get("offset", {})
                # VRM 0.x stores collider offsets in Unity's axes: flip Z for glTF.
                out.append(Collider(node, (float(o.get("x", 0)), float(o.get("y", 0)), -float(o.get("z", 0))),
                                    float(c.get("radius", 0))))
    return [c for c in out if c.node is not None and c.node >= 0 and c.radius > 0]


def compute_tangents(pos, nrm, uv, faces):
    tan = np.zeros_like(pos)
    bit = np.zeros_like(pos)
    p0, p1, p2 = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    t0, t1, t2 = uv[faces[:, 0]], uv[faces[:, 1]], uv[faces[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = t1 - t0, t2 - t0
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    det[np.abs(det) < 1e-12] = 1e-12
    r = 1.0 / det
    sdir = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r[:, None]
    tdir = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r[:, None]
    for k in range(3):
        np.add.at(tan, faces[:, k], sdir)
        np.add.at(bit, faces[:, k], tdir)
    tan = tan - nrm * np.sum(nrm * tan, axis=1, keepdims=True)
    length = np.linalg.norm(tan, axis=1, keepdims=True)
    fallback = np.cross(nrm, [0.0, 1.0, 0.0])
    bad = length[:, 0] < 1e-9
    tan[bad] = fallback[bad]
    length = np.linalg.norm(tan, axis=1, keepdims=True)
    length[length < 1e-9] = 1
    tan = tan / length
    sign = np.where(np.sum(np.cross(nrm, tan) * bit, axis=1) < 0, -1.0, 1.0)
    return np.hstack([tan, sign[:, None]])


def _face_normals_to_vertices(pos, faces):
    n = np.zeros_like(pos)
    fn = np.cross(pos[faces[:, 1]] - pos[faces[:, 0]], pos[faces[:, 2]] - pos[faces[:, 0]])
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    length[length == 0] = 1
    return n / length


def load_vrm(path):
    with open(path, "rb") as f:
        data = f.read()
    return load_vrm_bytes(data, os.path.dirname(os.path.abspath(path)))


def load_vrm_bytes(data, base_dir="."):
    js, binary = parse_glb(data)
    g = _Gltf(js, binary, base_dir)
    version, humanoid = _canonical_humanoid(js)
    nodes = js.get("nodes", [])
    parents = [-1] * len(nodes)
    for i, node in enumerate(nodes):
        for c in node.get("children", []):
            parents[c] = i
    local = np.array([_node_matrix(n) for n in nodes]) if nodes else np.zeros((0, 4, 4))
    world = np.zeros_like(local)
    done = [False] * len(nodes)

    def solve(i):
        if not done[i]:
            world[i] = local[i] if parents[i] < 0 else solve(parents[i]) @ local[i]
            done[i] = True
        return world[i]

    for i in range(len(nodes)):
        solve(i)

    prims = []
    for ni, node in enumerate(nodes):
        if "mesh" not in node:
            continue
        mesh = js["meshes"][node["mesh"]]
        skin = js["skins"][node["skin"]] if "skin" in node else None
        joint_mats = None
        if skin:
            joints = skin["joints"]
            if "inverseBindMatrices" in skin:
                ibm = g.accessor(skin["inverseBindMatrices"]).astype(np.float64).reshape(-1, 4, 4).transpose(0, 2, 1)
            else:
                ibm = np.tile(np.eye(4), (len(joints), 1, 1))
            joint_mats = np.array([world[j] @ ibm[k] for k, j in enumerate(joints)])
        for prim in mesh.get("primitives", []):
            if prim.get("mode", 4) != 4:
                continue
            attrs = prim["attributes"]
            if "POSITION" not in attrs:
                continue
            pos = g.accessor(attrs["POSITION"]).astype(np.float64)[:, :3]
            n = len(pos)
            if "indices" in prim:
                faces = g.accessor(prim["indices"]).astype(np.int64).reshape(-1, 3)
            else:
                faces = np.arange(n - n % 3, dtype=np.int64).reshape(-1, 3)
            nrm = g.accessor(attrs["NORMAL"]).astype(np.float64)[:, :3] if "NORMAL" in attrs else None
            tan = g.accessor(attrs["TANGENT"]).astype(np.float64)[:, :4] if "TANGENT" in attrs else None
            uv = g.accessor(attrs["TEXCOORD_0"]).astype(np.float64)[:, :2] if "TEXCOORD_0" in attrs else np.zeros((n, 2))

            if skin and "JOINTS_0" in attrs and "WEIGHTS_0" in attrs:
                jidx = g.accessor(attrs["JOINTS_0"]).astype(np.int64)[:, :4]
                wts = g.accessor(attrs["WEIGHTS_0"]).astype(np.float64)[:, :4]
                wsum = wts.sum(axis=1, keepdims=True)
                zero = wsum[:, 0] <= 1e-8
                wts[zero] = [1, 0, 0, 0]
                wsum[zero] = 1
                wts = wts / wsum
                jidx = np.clip(jidx, 0, len(skin["joints"]) - 1)
                blend = np.einsum("nk,nkij->nij", wts, joint_mats[jidx])
                node_joints = np.array(skin["joints"], dtype=np.int64)[jidx]
            else:
                blend = np.tile(world[ni], (n, 1, 1))
                wts = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
                attach = ni
                node_joints = np.tile([attach, attach, attach, attach], (n, 1))

            homo = np.hstack([pos, np.ones((n, 1))])
            wpos = np.einsum("nij,nj->ni", blend, homo)[:, :3]
            lin = blend[:, :3, :3]
            if nrm is None:
                wn = _face_normals_to_vertices(wpos, faces)
            else:
                inv_t = np.linalg.inv(lin).transpose(0, 2, 1)
                wn = np.einsum("nij,nj->ni", inv_t, nrm)
                ln = np.linalg.norm(wn, axis=1, keepdims=True)
                ln[ln == 0] = 1
                wn = wn / ln
            if tan is not None:
                wt = np.einsum("nij,nj->ni", lin, tan[:, :3])
                lt = np.linalg.norm(wt, axis=1, keepdims=True)
                lt[lt == 0] = 1
                wt = np.hstack([wt / lt, np.where(tan[:, 3:4] < 0, -1.0, 1.0)])
            else:
                wt = None
            prims.append(Primitive(wpos, wn, wt, uv, node_joints, wts, faces, prim.get("material", -1), ni))

    images = []
    for i in range(len(js.get("images", []))):
        try:
            images.append(g.image_bytes(i))
        except Exception:
            images.append(None)

    meta = {}
    ext = js.get("extensions", {})
    if "VRM" in ext:
        meta = ext["VRM"].get("meta", {})
    elif "VRMC_vrm" in ext:
        meta = ext["VRMC_vrm"].get("meta", {})
    title = meta.get("title") or meta.get("name") or ""
    springs = [c for c in _springs(js) if all(0 <= n < len(nodes) for n in c.nodes)]
    # Humanoid bones are animated by the game, never by physics.
    human_nodes = set(humanoid.values())
    springs = [SpringChain([n for n in c.nodes if n not in human_nodes], c.stiffness, c.gravity, c.drag, c.radius)
               for c in springs]
    springs = [c for c in springs if c.nodes]
    colliders = [c for c in _colliders(js) if c.node < len(nodes)]
    return VrmModel(version, nodes, parents, world, humanoid, prims, _materials(g), images, title, meta, springs,
                    colliders)
