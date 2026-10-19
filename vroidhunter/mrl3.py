"""Reader and writer for .mrl3 material lists (Iceborne layout).

Structure from CrazyT's 010 template and AsteriskAmpersand's material editor
research. Materials are cloned from the reference armor so shader parameters
stay valid; only names and texture slots are changed.
"""

import struct
from dataclasses import dataclass, field

from .binio import Reader, align, pad_to, jamcrc

MRL3_MAGIC = 0x4C524D  # "MRL\0"
HEADER_FMT = "<i12BIIQQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 0x28
TEX_FMT = "<I12B256s"
TEX_SIZE = struct.calcsize(TEX_FMT)  # 272
MAT_FMT = "<IIIIIhB9BB15BIi"
MAT_SIZE = struct.calcsize(MAT_FMT)  # 56
BIND_FMT = "<B3BIIi"
BIND_SIZE = struct.calcsize(BIND_FMT)  # 16

RES_CBUFFER, RES_SAMPLER, RES_TEXTURE = 0, 1, 2

# Texture slot names used by player equipment shaders. A slot's mapType stores
# (jamcrc(name) & 0xFFFFF) << 12, so names are matched by that partial hash.
SLOT_NAMES = [
    "tAlbedoMap", "tAlbedoMap__disclosure", "tBaseMap", "tAlbedoExtendMap__disclosure",
    "tAlbedoUniqueMap__disclosure", "tAlbedoOverMap__disclosure",
    "tNormalMap", "tNormalMap__disclosure", "tFaceNormalMap__disclosure", "tBakedNormalMap",
    "tDetailNormalMap__disclosure", "tAddNormalMap__disclosure", "tNormalHeightMap__disclosure",
    "tRMTMap", "tRMTMap__disclosure",
    "tEmissiveMap", "tEmissiveMap__disclosure", "tDetailEmissiveMap__disclosure",
    "tColorMaskMap__disclosure", "tPartsMaskMap__disclosure", "tMaskMap__disclosure",
    "tAlphaMap", "tAlphaMap__disclosure", "tOpacityMap__disclosure",
    "tFurMap__disclosure", "tFurMaskMap__disclosure", "tFurNormalMap__disclosure",
    "tFurBlendMap__disclosure", "tFurNormalBlendMap__disclosure", "tFurVelocityMap__disclosure",
    "tDetailMapA__disclosure", "tDetailMapB__disclosure", "tDetailMapC__disclosure",
    "tDetailMapD__disclosure", "tDetailNormalBlendMap__disclosure", "tFxMap__disclosure",
    "tPatternMap__disclosure", "tFacePaintMap__disclosure", "tFaceMayuMap__disclosure",
    "tEmissiveBlendMap__disclosure", "tAddNormalMaskMap__disclosure", "tMaterialBlendMap__disclosure",
]
SLOT_BY_HASH = {jamcrc(n) & 0xFFFFF: n for n in SLOT_NAMES}


def slot_role(map_type):
    """Classify a texture slot into the role our generated textures fill."""
    name = SLOT_BY_HASH.get(map_type >> 12, "")
    if not name:
        return "unknown", name
    if name.startswith("tAlbedo") or name == "tBaseMap":
        return "albedo", name
    if "Fur" in name:
        return "black", name
    if "NormalMap" in name or "NormalBlend" in name or "NormalHeight" in name:
        return "normal", name
    if name.startswith("tRMT"):
        return "rmt", name
    if "Emissive" in name or "ColorMask" in name or "PartsMask" in name or "MaskMap" in name:
        return "black", name
    if name.startswith("tAlpha") or name.startswith("tOpacity"):
        return "alpha", name
    if name.startswith("tDetailMap") or "Pattern" in name or "FacePaint" in name or "Mayu" in name \
            or "MaterialBlend" in name or name.startswith("tFx"):
        return "black", name
    return "unknown", name


@dataclass
class Binding:
    res_type: int
    unkn: tuple
    map_type: int
    tex_idx: int
    unkn2: int

    def pack(self):
        return struct.pack(BIND_FMT, self.res_type, *self.unkn, self.map_type, self.tex_idx, self.unkn2)


@dataclass
class Material:
    fields: list               # raw header field values (MAT_FMT order)
    bindings: list
    params: bytes              # constant buffer data following the bindings

    @property
    def name_hash(self):
        return self.fields[1]

    def clone(self):
        return Material(list(self.fields),
                        [Binding(b.res_type, tuple(b.unkn), b.map_type, b.tex_idx, b.unkn2) for b in self.bindings],
                        self.params)


@dataclass
class Mrl3:
    header: list
    textures: list            # list of (type_code, unkn12 tuple, path str)
    materials: list = field(default_factory=list)


def read_mrl3(data):
    r = Reader(data)
    head = list(r.unpack(HEADER_FMT))
    if head[0] != MRL3_MAGIC:
        raise ValueError("not a mrl3 file")
    mat_count, tex_count, tex_off, mat_off = head[13], head[14], head[15], head[16]
    r.seek(tex_off)
    textures = []
    for _ in range(tex_count):
        vals = r.unpack(TEX_FMT)
        textures.append((vals[0], tuple(vals[1:13]), vals[13].split(b"\x00", 1)[0].decode("ascii", "replace")))
    r.seek(mat_off)
    materials = []
    for _ in range(mat_count):
        vals = list(r.unpack(MAT_FMT))
        mat_size, float_off, start = vals[4], vals[6], vals[-2]
        nbind = (float_off * 8) // BIND_SIZE
        br = Reader(data, start)
        binds = []
        for _ in range(nbind):
            t, a, b, c, mt, ti, u = br.unpack(BIND_FMT)
            binds.append(Binding(t, (a, b, c), mt, ti, u))
        params = data[start + nbind * BIND_SIZE:start + mat_size]
        materials.append(Material(vals, binds, params))
    return Mrl3(head, textures, materials)


def write_mrl3(mrl):
    head = list(mrl.header)
    head[13] = len(mrl.materials)
    head[14] = len(mrl.textures)
    head[15] = HEADER_SIZE
    head[16] = HEADER_SIZE + TEX_SIZE * len(mrl.textures)
    out = bytearray(struct.pack(HEADER_FMT, *head))
    for code, unkn, path in mrl.textures:
        raw = path.encode("ascii")
        if len(raw) >= 256:
            raise ValueError("texture path too long: " + path)
        out += struct.pack(TEX_FMT, code, *unkn, raw.ljust(256, b"\x00"))

    blobs = []
    for m in mrl.materials:
        blob = b"".join(b.pack() for b in m.bindings) + m.params
        blobs.append(blob)
    position = align(head[16] + MAT_SIZE * len(mrl.materials))
    starts = []
    for blob in blobs:
        starts.append(position)
        position = align(position + len(blob))
    for m, blob, start in zip(mrl.materials, blobs, starts):
        vals = list(m.fields)
        vals[4] = len(blob)
        vals[6] = (len(m.bindings) * BIND_SIZE) // 8
        vals[-2] = start
        out += struct.pack(MAT_FMT, *vals)
    for blob, start in zip(blobs, starts):
        pad_to(out, start)
        out += blob
    pad_to(out, align(len(out)))
    return bytes(out)
