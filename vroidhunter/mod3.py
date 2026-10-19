"""Reader and writer for MT Framework .mod3 models as used by Monster Hunter: World.

The layout follows the community research by CrazyT, PredatorCZ and
AsteriskAmpersand (see THIRD_PARTY_NOTICES.md). Only what this tool needs is
implemented: the skeleton, material names and mesh headers are read from a
reference model, and new skinned meshes are written next to that skeleton.
"""

import struct
from dataclasses import dataclass, field

import numpy as np

from .binio import Reader, align, pad_to, jamcrc

MOD3_MAGIC = 0x444F4D  # "MOD\0"

HEADER_FMT = "<iBBHHHiiiiiQQQQQQQQQQ"
HEADER_BASE_SIZE = struct.calcsize(HEADER_FMT)  # 112
HEADER_EXTRA_SIZE = 48 + 96 + 64  # boundaries, float segment, byte segment
HEADER_SIZE = HEADER_BASE_SIZE + HEADER_EXTRA_SIZE  # 0x140

BONE_FMT = "<hBBfffff"
BONE_SIZE = struct.calcsize(BONE_FMT)  # 24
MATRIX_SIZE = 64
REMAP_SIZE = 512
GROUP_SIZE = 32
MATERIAL_NAME_SIZE = 128

MESH_FMT = "<HHHHiHBBIIIIIIBBHHHhhiiI4i"
MESH_SIZE = struct.calcsize(MESH_FMT)  # 80
MESH_FIELDS = (
    "shadowCast", "vertexCount", "visibleCondition", "materialIdx", "lod",
    "weightDynamics", "blockSize", "unkn3", "vertexSub", "vertexOffset",
    "blocktype", "faceOffset", "faceCount", "vertexBase", "NULL_0",
    "boundingBoxCount", "unknownIndex", "vertexSubMirror", "vertexIndexSub",
    "mapData0", "mapData1", "NULL_1", "intUnknown", "vertexSubTotal",
    "NULL_2a", "NULL_2b", "NULL_2c", "NULL_2d",
)

BBOX_FMT = "<i12B3ff4f4f16f4f"
BBOX_SIZE = struct.calcsize(BBOX_FMT)  # 144

# Vertex buffer types (hash -> description). Only the skinned 1-UV ones are written.
BLOCK_SKIN4_1UV = 0xF637401C
BLOCK_SKIN8_1UV = 0x81F58067
BLOCK_TYPES = {
    0xA756F2F9: dict(uvs=1, weights=0, colour=False),
    0xA5104CA0: dict(uvs=2, weights=0, colour=False),
    0xC3AC03A1: dict(uvs=3, weights=0, colour=True),
    0xC9690AB8: dict(uvs=4, weights=0, colour=True),
    0x818904DC: dict(uvs=1, weights=0, colour=True),
    0x0F06033F: dict(uvs=2, weights=0, colour=True),
    0xF637401C: dict(uvs=1, weights=4, colour=False),
    0xF471FE45: dict(uvs=2, weights=4, colour=False),
    0x3C730760: dict(uvs=1, weights=4, colour=True),
    0xB2FC0083: dict(uvs=2, weights=4, colour=True),
    0x81F58067: dict(uvs=1, weights=8, colour=False),
    0x83B33E3E: dict(uvs=2, weights=8, colour=False),
    0x366995A7: dict(uvs=1, weights=8, colour=True),
    0xB8E69244: dict(uvs=2, weights=8, colour=True),
}


def block_size(blocktype):
    info = BLOCK_TYPES[blocktype]
    size = 12 + 4 + 4 + 4 * info["uvs"]
    if info["weights"] == 4:
        size += 4 + 4
    elif info["weights"] == 8:
        size += 8 + 8
    if info["colour"]:
        size += 4
    return size


def _matrix_from_floats(values):
    """mod3 stores 4x4 matrices column by column; return a row-major numpy matrix."""
    return np.array(values, dtype=np.float64).reshape(4, 4).T


@dataclass
class Bone:
    function: int
    parent: int
    child: int
    unkn2: float
    length: float
    position: tuple


@dataclass
class Mod3:
    header: dict
    header_extra: bytes
    bones: list
    lmatrices: np.ndarray
    amatrices: np.ndarray
    remap: bytes
    skeleton_raw: bytes
    groups_raw: bytes
    materials: list
    meshes: list
    trailing: bytes
    data: bytes = field(repr=False, default=b"")

    # ---- skeleton helpers -------------------------------------------------
    def bone_world_matrices(self):
        """World (bind) matrix of every bone, row-major, translation in column 3."""
        n = len(self.bones)
        worlds = np.zeros((n, 4, 4))
        done = [False] * n

        def solve(i):
            if done[i]:
                return worlds[i]
            parent = self.bones[i].parent
            if parent == 255 or parent >= n:
                worlds[i] = self.lmatrices[i]
            else:
                worlds[i] = solve(parent) @ self.lmatrices[i]
            done[i] = True
            return worlds[i]

        for i in range(n):
            # AMatrix is documented as the inverse of the bone's absolute matrix.
            try:
                inv = np.linalg.inv(self.amatrices[i])
                if np.all(np.isfinite(inv)) and abs(inv[3, 3] - 1.0) < 1e-3:
                    worlds[i] = inv
                    done[i] = True
                    continue
            except np.linalg.LinAlgError:
                pass
        for i in range(n):
            solve(i)
        return worlds

    def function_to_index(self):
        """Map bone function id -> skeleton index."""
        out = {}
        for i, bone in enumerate(self.bones):
            if bone.function not in out:
                out[bone.function] = i
        return out

    # ---- mesh helpers -----------------------------------------------------
    def mesh_geometry(self, mesh_index):
        """Positions, normals and local triangle indices of one reference mesh."""
        h = self.meshes[mesh_index]
        if h["blocktype"] not in BLOCK_TYPES or h["vertexCount"] == 0:
            return None
        stride = h["blockSize"]
        start = self.header["vertexOffset"] + h["vertexOffset"] + stride * (h["vertexSub"] + h["vertexBase"])
        count = h["vertexCount"]
        raw = self.data[start:start + stride * count]
        if len(raw) < stride * count:
            return None
        verts = np.frombuffer(raw, dtype=np.uint8).reshape(count, stride)
        pos = verts[:, 0:12].copy().view("<f4").reshape(count, 3).astype(np.float64)
        nrm = verts[:, 12:15].copy().view(np.int8).astype(np.float64)
        fstart = self.header["facesOffset"] + h["faceOffset"] * 2
        fraw = self.data[fstart:fstart + 2 * h["faceCount"]]
        faces = np.frombuffer(fraw, dtype="<u2").astype(np.int64)
        faces = faces[: (len(faces) // 3) * 3].reshape(-1, 3) - h["vertexSub"]
        faces = faces[np.all((faces >= 0) & (faces < count), axis=1)]
        return pos, nrm, faces


def _mesh_skin(self, mesh_index):
    """(positions, bone indices (n,4), weights (n,4)) of a skinned reference mesh, or None."""
    h = self.meshes[mesh_index]
    info = BLOCK_TYPES.get(h["blocktype"])
    if info is None or not info["weights"] or h["vertexCount"] == 0:
        return None
    stride = h["blockSize"]
    start = self.header["vertexOffset"] + h["vertexOffset"] + stride * (h["vertexSub"] + h["vertexBase"])
    count = h["vertexCount"]
    raw = self.data[start:start + stride * count]
    if len(raw) < stride * count:
        return None
    v = np.frombuffer(raw, dtype=np.uint8).reshape(count, stride)
    pos = v[:, 0:12].copy().view("<f4").reshape(count, 3).astype(np.float64)
    w_at = 20 + 4 * info["uvs"]
    packed = v[:, w_at:w_at + 4].copy().view("<u4").ravel()
    w = np.stack([(packed >> (10 * k)) & 1023 for k in range(3)], axis=1) / 1023.0
    if info["weights"] == 4:
        w = np.hstack([w, np.clip(1 - w.sum(axis=1, keepdims=True), 0, 1)])
        ids = v[:, w_at + 4:w_at + 8].astype(np.int64)
    else:
        extra = v[:, w_at + 4:w_at + 7].astype(np.float64) / 255.0
        w = np.hstack([w, extra[:, :1]])
        ids = v[:, w_at + 8:w_at + 12].astype(np.int64)
    total = w.sum(axis=1, keepdims=True)
    total[total <= 0] = 1
    return pos, ids, w / total


Mod3.mesh_skin = _mesh_skin


def read_mod3(data):
    r = Reader(data)
    vals = r.unpack(HEADER_FMT)
    names = ("id", "version", "version2", "boneCount", "meshCount", "materialCount",
             "vertexCount", "faceCount", "vertexIds", "vertexBufferSize", "secondBufferSize",
             "groupCount", "creationDate", "boneOffset", "groupOffset", "materialNamesOffset",
             "meshOffset", "vertexOffset", "facesOffset", "trailOffset", "unkOffset2")
    header = dict(zip(names, vals))
    if header["id"] != MOD3_MAGIC:
        raise ValueError("not a mod3 file")
    header_extra = r.read(HEADER_EXTRA_SIZE)

    nb = header["boneCount"]
    bones, lm, am, remap, skeleton_raw = [], np.zeros((nb, 4, 4)), np.zeros((nb, 4, 4)), b"", b""
    if nb:
        r.seek(header["boneOffset"])
        for _ in range(nb):
            f, p, c, u2, ln, x, y, z = r.unpack(BONE_FMT)
            bones.append(Bone(f, p, c, u2, ln, (x, y, z)))
        for i in range(nb):
            lm[i] = _matrix_from_floats(r.unpack("<16f"))
        for i in range(nb):
            am[i] = _matrix_from_floats(r.unpack("<16f"))
        remap = r.read(REMAP_SIZE)
        skel_len = nb * (BONE_SIZE + 2 * MATRIX_SIZE) + REMAP_SIZE
        skeleton_raw = data[header["boneOffset"]:header["boneOffset"] + skel_len]

    groups_raw = data[header["groupOffset"]:header["groupOffset"] + GROUP_SIZE * header["groupCount"]]

    r.seek(header["materialNamesOffset"])
    materials = []
    for _ in range(header["materialCount"]):
        materials.append(r.read(MATERIAL_NAME_SIZE).split(b"\x00", 1)[0].decode("ascii", "replace"))

    r.seek(header["meshOffset"])
    meshes = []
    for _ in range(header["meshCount"]):
        meshes.append(dict(zip(MESH_FIELDS, r.unpack(MESH_FMT))))

    trailing = data[header["trailOffset"]:] if header["trailOffset"] else b""
    return Mod3(header, header_extra, bones, lm, am, remap, skeleton_raw, groups_raw,
                materials, meshes, trailing, data)


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

@dataclass
class MeshPart:
    """One mesh to write. All arrays are per vertex unless noted."""
    material_index: int
    positions: np.ndarray   # (n,3) float
    normals: np.ndarray     # (n,3) float, unit length
    tangents: np.ndarray    # (n,4) float, xyz + handedness sign
    uvs: np.ndarray         # (n,2) float, mod3 convention (v down, same as glTF)
    bone_ids: np.ndarray    # (n,4) skeleton indices
    weights: np.ndarray     # (n,4) float, rows sum to 1, sorted descending
    faces: np.ndarray       # (m,3) local vertex indices


def _encode_normals(n):
    n = np.asarray(n, dtype=np.float64)
    m = np.max(np.abs(n), axis=1, keepdims=True)
    m[m == 0] = 1.0
    out = np.zeros((len(n), 4), dtype=np.int8)
    out[:, :3] = np.clip(np.round(127.0 * n / m), -127, 127).astype(np.int8)
    return out


def _encode_tangents(t):
    t = np.asarray(t, dtype=np.float64)
    out = np.zeros((len(t), 4), dtype=np.int8)
    out[:, :3] = np.clip(np.round(127.0 * t[:, :3]), -127, 127).astype(np.int8)
    out[:, 3] = np.where(t[:, 3] < 0, -127, 127).astype(np.int8)
    return out


def _quantize_weights(weights, levels, count):
    """Quantize the first `count` weights so their total never exceeds `levels`."""
    q = np.round(np.asarray(weights[:, :count]) * levels).astype(np.int64)
    q = np.clip(q, 0, levels)
    over = q.sum(axis=1) - levels
    for i in np.nonzero(over > 0)[0]:
        excess = over[i]
        for j in range(count - 1, -1, -1):
            take = min(excess, q[i, j])
            q[i, j] -= take
            excess -= take
            if excess == 0:
                break
    return q


def pack_vertices(part, blocktype):
    n = len(part.positions)
    stride = block_size(blocktype)
    buf = np.zeros((n, stride), dtype=np.uint8)
    buf[:, 0:12] = np.asarray(part.positions, dtype="<f4").reshape(n, 3).view(np.uint8).reshape(n, 12)
    buf[:, 12:16] = _encode_normals(part.normals).view(np.uint8)
    buf[:, 16:20] = _encode_tangents(part.tangents).view(np.uint8)
    buf[:, 20:24] = np.asarray(part.uvs, dtype="<f2").reshape(n, 2).view(np.uint8).reshape(n, 4)
    weights = np.asarray(part.weights, dtype=np.float64)
    ids = np.asarray(part.bone_ids, dtype=np.int64)
    if BLOCK_TYPES[blocktype]["weights"] == 4:
        q = _quantize_weights(weights, 1023, 3)
        packed = (q[:, 0] | (q[:, 1] << 10) | (q[:, 2] << 20)).astype("<u4")
        buf[:, 24:28] = packed.view(np.uint8).reshape(n, 4)
        buf[:, 28:32] = np.clip(ids[:, :4], 0, 255).astype(np.uint8)
    elif BLOCK_TYPES[blocktype]["weights"] == 8:
        w8 = np.zeros((n, 8))
        w8[:, :4] = weights[:, :4]
        q = _quantize_weights(w8, 1023, 3)
        packed = (q[:, 0] | (q[:, 1] << 10) | (q[:, 2] << 20)).astype("<u4")
        buf[:, 24:28] = packed.view(np.uint8).reshape(n, 4)
        rest = np.clip(np.round(w8[:, 3:7] * 255), 0, 255).astype(np.uint8)
        buf[:, 28:32] = rest
        ids8 = np.zeros((n, 8), dtype=np.uint8)
        ids8[:, :4] = np.clip(ids[:, :4], 0, 255)
        buf[:, 32:40] = ids8
    else:
        raise ValueError("unsupported block type for skinned output")
    return buf.tobytes()


def _bounding_boxes(part, bone_inverse_worlds):
    """One box per skeleton bone that the mesh is weighted to, in that bone's space."""
    boxes = []
    pos = np.asarray(part.positions, dtype=np.float64)
    homo = np.hstack([pos, np.ones((len(pos), 1))])
    ids = np.asarray(part.bone_ids)
    w = np.asarray(part.weights)
    for bone in sorted(set(int(b) for b in ids[w > 0].ravel())):
        mask = np.any((ids == bone) & (w > 0), axis=1)
        if not np.any(mask):
            continue
        local = (bone_inverse_worlds[bone] @ homo[mask].T).T[:, :3]
        lo, hi = local.min(axis=0), local.max(axis=0)
        center = (lo + hi) / 2.0
        half = np.maximum((hi - lo) / 2.0, 0.01)
        lo, hi = center - half, center + half
        radius = float(np.linalg.norm(half))
        matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, center[0], center[1], center[2], 1]
        boxes.append(struct.pack(
            BBOX_FMT, bone, *([0xCD] * 12), *center, radius,
            *lo, 0.0, *hi, 0.0, *matrix, *half, 0.0))
    return boxes


def write_mod3(ref, parts, material_names, blocktype=BLOCK_SKIN4_1UV, mesh_template=None):
    """Write a new mod3 that keeps `ref`'s skeleton, groups and header data.

    parts must each have at most 65535 vertices.
    mesh_template: header dict whose flag fields (shadowCast, weightDynamics, ...)
    are copied into every written mesh.
    """
    if mesh_template is None:
        mesh_template = {}
    stride = block_size(blocktype)
    worlds = ref.bone_world_matrices() if ref.bones else np.zeros((0, 4, 4))
    inv_worlds = np.array([np.linalg.inv(m) for m in worlds]) if len(worlds) else worlds

    mesh_headers, bbox_blobs, vertex_blobs, face_blobs = [], [], [], []
    v_count = v_len = prev_base = prev_sub = prev_bs = 0
    f_offset = 0
    total_faces = 0
    edges = set()
    for part in parts:
        n = len(part.positions)
        if n > 0xFFFF:
            raise ValueError("mesh part has more than 65535 vertices")
        h = {k: 0 for k in MESH_FIELDS}
        for key in ("shadowCast", "visibleCondition", "weightDynamics", "unkn3",
                    "unknownIndex", "mapData0", "mapData1", "intUnknown"):
            if key in mesh_template:
                h[key] = mesh_template[key]
        if "shadowCast" not in mesh_template:
            h["shadowCast"] = 0x13
        if "weightDynamics" not in mesh_template:
            h["weightDynamics"] = 25
        if "mapData0" not in mesh_template:
            h["mapData0"] = h["mapData1"] = -1
        h["vertexCount"] = n
        h["materialIdx"] = part.material_index
        h["lod"] = 0xFFFF
        h["blockSize"] = stride
        h["blocktype"] = blocktype

        # Vertex base / sub / offset bookkeeping (see Mod3-Structure-Notes).
        if stride != prev_bs:
            h["vertexOffset"] = v_len
            sub, base = 0, 0
        else:
            if n + prev_sub > 0xFFFF:
                sub, base = 0, prev_base + prev_sub
            else:
                sub, base = prev_sub, prev_base
            h["vertexOffset"] = v_len - (base + sub) * stride
        h["vertexSub"] = sub
        h["vertexBase"] = base
        h["vertexSubMirror"] = sub
        h["vertexSubTotal"] = v_count
        h["vertexIndexSub"] = (sub + n - 1) & 0xFFFF
        v_count, v_len, prev_base, prev_sub, prev_bs = v_count + n, v_len + n * stride, base, sub + n, stride

        faces = np.asarray(part.faces, dtype=np.int64).reshape(-1, 3)
        if len(faces) and (faces.max() >= n or faces.min() < 0):
            raise ValueError("face index out of range")
        idx = (faces + sub).astype("<u2")
        h["faceOffset"] = f_offset // 2
        h["faceCount"] = faces.size
        f_offset += idx.nbytes
        total_faces += faces.size
        for a, b in ((0, 1), (1, 2), (2, 0)):
            e = np.sort(faces[:, [a, b]] + v_count - n, axis=1)
            edges.update(map(tuple, e.tolist()))

        boxes = _bounding_boxes(part, inv_worlds) if len(inv_worlds) else []
        if len(boxes) > 255:
            boxes = boxes[:255]
        h["boundingBoxCount"] = len(boxes)
        bbox_blobs.extend(boxes)
        mesh_headers.append(h)
        vertex_blobs.append(pack_vertices(part, blocktype))
        face_blobs.append(idx.tobytes())

    out = bytearray()
    header = dict(ref.header)
    header["boneCount"] = len(ref.bones)
    header["meshCount"] = len(parts)
    header["materialCount"] = len(material_names)
    header["vertexCount"] = v_count
    header["faceCount"] = total_faces
    header["vertexIds"] = len(edges)
    header["vertexBufferSize"] = v_len
    header["secondBufferSize"] = 0
    header["unkOffset2"] = 0

    skeleton = ref.skeleton_raw if ref.bones else b""
    groups = ref.groups_raw
    header["groupCount"] = len(groups) // GROUP_SIZE
    mat_blob = b"".join(name.encode("ascii").ljust(MATERIAL_NAME_SIZE, b"\x00")[:MATERIAL_NAME_SIZE - 1] + b"\x00"
                        for name in material_names)
    for name in material_names:
        if len(name) >= MATERIAL_NAME_SIZE:
            raise ValueError("material name too long: " + name)
    mesh_blob = b"".join(struct.pack(MESH_FMT, *[h[k] for k in MESH_FIELDS]) for h in mesh_headers)
    bbox_blob = struct.pack("<I", len(bbox_blobs)) + b"".join(bbox_blobs)
    vert_blob = b"".join(vertex_blobs)
    face_blob = b"".join(face_blobs)

    current = HEADER_SIZE
    header["boneOffset"] = align(current) if ref.bones else 0
    if ref.bones:
        current = header["boneOffset"] + len(skeleton)
    header["groupOffset"] = align(current)
    header["materialNamesOffset"] = align(header["groupOffset"] + len(groups))
    header["meshOffset"] = align(header["materialNamesOffset"] + len(mat_blob))
    header["vertexOffset"] = header["meshOffset"] + len(mesh_blob) + len(bbox_blob)
    header["facesOffset"] = header["vertexOffset"] + len(vert_blob)
    header["trailOffset"] = align(header["facesOffset"] + len(face_blob), 4)

    # Model-space bounds in the header "boundaries" segment.
    extra = bytearray(ref.header_extra)
    all_pos = np.vstack([p.positions for p in parts]) if parts else np.zeros((1, 3))
    lo, hi = all_pos.min(axis=0), all_pos.max(axis=0)
    center = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo) / 2)
    extra[0:48] = struct.pack("<3ff4f4f", *center, radius, *lo, 0.0, *hi, 0.0)

    names = ("id", "version", "version2", "boneCount", "meshCount", "materialCount",
             "vertexCount", "faceCount", "vertexIds", "vertexBufferSize", "secondBufferSize",
             "groupCount", "creationDate", "boneOffset", "groupOffset", "materialNamesOffset",
             "meshOffset", "vertexOffset", "facesOffset", "trailOffset", "unkOffset2")
    out += struct.pack(HEADER_FMT, *[header[k] for k in names])
    out += extra
    if ref.bones:
        pad_to(out, header["boneOffset"])
        out += skeleton
    pad_to(out, header["groupOffset"])
    out += groups
    pad_to(out, header["materialNamesOffset"])
    out += mat_blob
    pad_to(out, header["meshOffset"])
    out += mesh_blob + bbox_blob + vert_blob + face_blob
    pad_to(out, header["trailOffset"])
    out += ref.trailing
    return bytes(out)


def blank_part(ref, material_index=0):
    """A single degenerate triangle: keeps a model valid while drawing nothing."""
    n = 3
    root = 0
    pos = np.zeros((n, 3))
    worlds = ref.bone_world_matrices()
    if len(worlds):
        pos[:] = worlds[root][:3, 3]
    return MeshPart(
        material_index=material_index,
        positions=pos,
        normals=np.tile([0.0, 0.0, 1.0], (n, 1)),
        tangents=np.tile([1.0, 0.0, 0.0, 1.0], (n, 1)),
        uvs=np.zeros((n, 2)),
        bone_ids=np.zeros((n, 4), dtype=np.int64),
        weights=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        faces=np.array([[0, 1, 2]]),
    )


def material_hash(name):
    return jamcrc(name)


def _colmajor(m):
    return np.asarray(m, dtype=np.float64).T.reshape(16)


def extend_skeleton(ref, new_bones):
    """Copy of `ref` with extra bones appended.

    new_bones: list of (function, parent_function, world_position). Parents must be
    existing bones or bones earlier in the list. New bones get identity orientation.
    """
    import dataclasses
    if not new_bones:
        return ref
    worlds = list(ref.bone_world_matrices())
    bones = list(ref.bones)
    lms = list(ref.lmatrices)
    ams = list(ref.amatrices)
    index = ref.function_to_index()
    for func, parent_func, pos in new_bones:
        if func in index:
            raise ValueError("bone function %d already exists" % func)
        if not 0 <= func < REMAP_SIZE:
            raise ValueError("bone function %d out of range" % func)
        p = index[parent_func]
        world = np.eye(4)
        world[:3, 3] = pos
        local = np.linalg.inv(worlds[p]) @ world
        t = local[:3, 3]
        bones.append(Bone(func, p, 255, bones[p].unkn2, float(np.linalg.norm(t)), tuple(map(float, t))))
        lms.append(local)
        ams.append(np.linalg.inv(world))
        worlds.append(world)
        index[func] = len(bones) - 1
    if len(bones) > 255:
        raise ValueError("too many bones (%d, the limit is 255)" % len(bones))
    raw, remap = _serialize_skeleton(ref, bones, lms, ams)
    header = dict(ref.header)
    header["boneCount"] = len(bones)
    return dataclasses.replace(ref, header=header, bones=bones, lmatrices=np.array(lms),
                               amatrices=np.array(ams), remap=bytes(remap), skeleton_raw=bytes(raw))


def _serialize_skeleton(ref, bones, lms, ams):
    raw = bytearray()
    for b in bones:
        raw += struct.pack(BONE_FMT, b.function, b.parent, b.child, b.unkn2, b.length, *b.position)
    for m in lms:
        raw += struct.pack("<16f", *_colmajor(m))
    for m in ams:
        raw += struct.pack("<16f", *_colmajor(m))
    remap = bytearray(ref.remap) if ref.remap else bytearray([255] * REMAP_SIZE)
    for i, b in enumerate(bones):
        if b.function < REMAP_SIZE:
            remap[b.function] = i if i < 255 else 255
    raw += remap
    return bytes(raw), bytes(remap)


def repose_skeleton(ref, targets):
    """Copy of `ref` whose joints sit at new world positions.

    targets: {bone function: world position}. Those bones keep their world orientation
    and move; every other bone keeps its offset from its parent and follows along.
    """
    import dataclasses
    if not ref.bones or not targets:
        return ref
    old = ref.bone_world_matrices()
    n = len(ref.bones)
    new = [None] * n

    def solve(i):
        if new[i] is None:
            b = ref.bones[i]
            parent = b.parent if b.parent != 255 and b.parent < n else None
            if b.function in targets:
                w = old[i].copy()
                w[:3, 3] = targets[b.function]
            elif parent is None:
                w = old[i].copy()
            else:
                w = solve(parent) @ (np.linalg.inv(old[parent]) @ old[i])
            new[i] = w
        return new[i]

    for i in range(n):
        solve(i)
    bones, lms, ams = [], [], []
    for i, b in enumerate(ref.bones):
        parent = b.parent if b.parent != 255 and b.parent < n else None
        local = new[i] if parent is None else np.linalg.inv(new[parent]) @ new[i]
        t = local[:3, 3]
        bones.append(Bone(b.function, b.parent, b.child, b.unkn2, float(np.linalg.norm(t)), tuple(map(float, t))))
        lms.append(local)
        ams.append(np.linalg.inv(new[i]))
    raw, remap = _serialize_skeleton(ref, bones, lms, ams)
    return dataclasses.replace(ref, bones=bones, lmatrices=np.array(lms), amatrices=np.array(ams),
                               remap=remap, skeleton_raw=raw)
