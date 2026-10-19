"""Turn a VRM file into the nativePC files that replace one armor set."""

import io
import re
from dataclasses import dataclass, field

import numpy as np

from . import ccl as CL
from . import ctc as C
from . import mod3 as M
from . import mrl3 as R
from . import tex as X
from .fit import HunterSkeleton, fit_vrm
from .vrm import compute_tangents, load_vrm

# A few well known sets for the picker; any plXXX_YYYY folder in the game works.
KNOWN_ARMOR = {
    "pl001_0000": "Leather",
    "pl002_0000": "Hunter's",
    "pl003_0000": "Bone",
    "pl004_0000": "Alloy",
    "pl014_0000": "Chainmail",
    "pl048_0000": "Guild Cross",
    "pl053_0000": "Commission",
    "pl092_0000": "Guildwork",
    "pl094_0000": "Guild Palace",
    "pl121_0000": "Defender",
}
DEFAULT_ARMOR = "pl001_0000"
PARTS = ("helm", "body", "arm", "wst", "leg")
MAX_VERTS = 0xFFFF


@dataclass
class Options:
    armor: str = DEFAULT_ARMOR
    genders: tuple = ("f", "m")
    height_scale: float = 1.0
    physics: bool = True
    keep_proportions: bool = True   # False: stretch the avatar onto the hunter's body shape
    cover_neck: bool = True         # thicken the avatar's neck where the hunter's own neck would show
    max_texture: int = 2048
    double_sided: bool = True
    alpha_cull: bool = True


@dataclass
class BuildResult:
    files: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def armor_sets(game):
    """Armor folder ids available for both genders' equipment."""
    found = set()
    for g in ("f", "m"):
        for path in game.listdir("pl\\%s_equip\\" % g):
            m = re.match(r"pl\\[fm]_equip\\(pl\d{3}_\d{4})\\", path, re.IGNORECASE)
            if m and path.lower().endswith(".mod3"):
                found.add(m.group(1).lower())
    return sorted(found)


def _armor_files(game, gender, armor):
    prefix = "pl\\%s_equip\\%s\\" % (gender, armor)
    files = game.listdir(prefix)
    mods = [f for f in files if f.lower().endswith(".mod3")]
    mrls = [f for f in files if f.lower().endswith(".mrl3")]
    return mods, mrls


def _part_of(path):
    parts = path.lower().split("\\")
    for p in parts:
        if p in PARTS:
            return p
    return "other"


def _pick_reference_mesh(ref):
    """Index of the reference mesh with the most vertices at the highest detail."""
    best, best_n = None, -1
    for i, h in enumerate(ref.meshes):
        if h["blocktype"] not in M.BLOCK_TYPES or M.BLOCK_TYPES[h["blocktype"]]["weights"] == 0:
            continue
        lod = h["lod"]
        top = lod == 0xFFFF or (lod & 1) or lod == 0
        score = h["vertexCount"] + (1 << 20 if top else 0)
        if score > best_n:
            best, best_n = i, score
    return best


def _winding_sign(pos, nrm, faces):
    if len(faces) == 0:
        return 0.0
    p0, p1, p2 = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    vn = nrm[faces[:, 0]] + nrm[faces[:, 1]] + nrm[faces[:, 2]]
    return float(np.sum(np.sign(np.sum(fn * vn, axis=1))))


def _reference_winding(ref):
    total = 0.0
    for i in range(len(ref.meshes)):
        geo = ref.mesh_geometry(i)
        if geo is not None:
            total += _winding_sign(*geo)
    return 1.0 if total >= 0 else -1.0


def _hunter_skeleton(ref):
    worlds = ref.bone_world_matrices()
    pos, parents = {}, {}
    for i, b in enumerate(ref.bones):
        if b.function in pos:
            continue
        pos[b.function] = worlds[i][:3, 3].copy()
        p = b.parent
        parents[b.function] = ref.bones[p].function if p != 255 and p < len(ref.bones) else None
    return HunterSkeleton(pos, parents)


# ---------------------------------------------------------------------------
# Textures
# ---------------------------------------------------------------------------

def _open_image(data):
    from PIL import Image
    img = Image.open(io.BytesIO(data))
    img.load()
    return img.convert("RGBA")


def _resized(img, max_size):
    from PIL import Image
    w, h = X.fit_size(img.width, img.height, max_size)
    if (w, h) != (img.width, img.height):
        img = img.resize((w, h), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8).copy()


class TextureSet:
    """Collects generated textures; returns mrl3-style paths."""

    def __init__(self, folder, with_ext):
        self.folder = folder.rstrip("\\")
        self.with_ext = with_ext
        self.files = {}

    def add(self, name, rgba, fmt):
        path = self.folder + "\\" + name
        if path + ".tex" not in self.files:
            self.files[path + ".tex"] = X.write_tex(rgba, fmt)
        return path + (".tex" if self.with_ext else "")

    def solid(self, name, rgba):
        return self.add(name, np.tile(np.array(rgba, dtype=np.uint8), (8, 8, 1)), X.FMT_RGBA8)


@dataclass
class MaterialTextures:
    albedo: str
    normal: str
    rmt: str
    black: str
    alpha: str
    alpha_pixels: object  # (h,w) alpha array of the albedo, or None
    kind: str = "cloth"


def material_kind(name):
    """Rough surface type of a VRM material, from its name (VRoid names are consistent)."""
    n = (name or "").upper()
    if "HAIR" in n:
        return "hair"
    if "EYE" in n and "BROW" not in n and "LASH" not in n:
        return "eye"
    if "SKIN" in n or "FACE" in n or ("BODY" in n and "CLOTH" not in n):
        return "skin"
    return "cloth"


# Roughness / metal / translucency / subsurface for the RMT texture, per surface type.
KIND_RMT = {"skin": (150, 0, 0, 40), "hair": (120, 0, 0, 0), "eye": (90, 0, 0, 0), "cloth": (190, 0, 0, 0)}

# VRoid colours are authored for toon shading, where white means "fully lit". Physically
# based lighting needs albedo below ~85%, or bright areas glow in daylight.
ALBEDO_SCALE = 0.85


def _albedo_lut():
    x = np.arange(256) / 255.0
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4) * ALBEDO_SCALE
    out = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.clip(np.round(out * 255), 0, 255).astype(np.uint8)


_LUT = _albedo_lut()


def _material_textures(model, mat_index, texset, opts, cache):
    if mat_index in cache:
        return cache[mat_index]
    mat = model.materials[mat_index] if 0 <= mat_index < len(model.materials) else None
    tag = "vrh%02d" % max(mat_index, 0)
    color = np.array(mat.color if mat else (1, 1, 1, 1), dtype=np.float64)
    rgba = None
    if mat is not None and mat.base_image is not None and model.images[mat.base_image]:
        try:
            rgba = _resized(_open_image(model.images[mat.base_image]), opts.max_texture)
        except Exception:
            rgba = None
    if rgba is None:
        rgba = np.full((8, 8, 4), 255, dtype=np.uint8)
    if not np.allclose(color, 1.0):
        rgba = np.clip(rgba.astype(np.float64) * color[None, None, :], 0, 255).astype(np.uint8)
    rgba[:, :, :3] = _LUT[rgba[:, :, :3]]
    transparent = mat is not None and mat.alpha_mode != "OPAQUE"
    if transparent:
        albedo = texset.add(tag + "_BML", rgba, X.FMT_RGBA8_SRGB)
    else:
        rgba[:, :, 3] = 255
        albedo = texset.add(tag + "_BML", rgba, X.FMT_BC1_SRGB)

    normal = texset.solid("vrh_flat_NM", (128, 128, 0, 255))
    if mat is not None and mat.normal_image is not None and model.images[mat.normal_image]:
        try:
            nm = _resized(_open_image(model.images[mat.normal_image]), min(opts.max_texture, 1024))
            out = np.zeros_like(nm)
            out[:, :, 0] = nm[:, :, 0]
            out[:, :, 1] = 255 - nm[:, :, 1]   # MHW uses -Y (DirectX style) normal maps
            out[:, :, 3] = 255                 # blue left black, as MHW normal maps do
            normal = texset.add(tag + "_NM", out, X.FMT_RGBA8)
        except Exception:
            pass
    kind = material_kind(mat.name if mat else "")
    rmt = texset.solid("vrh_%s_RMT" % kind, KIND_RMT[kind])
    black = texset.solid("vrh_black", (0, 0, 0, 255))
    alpha_tex = texset.solid("vrh_white", (255, 255, 255, 255))
    if transparent:
        a = rgba[:, :, 3]
        alpha_tex = texset.add(tag + "_ALP", np.dstack([a, a, a, np.full_like(a, 255)]), X.FMT_RGBA8)
    result = MaterialTextures(albedo, normal, rmt, black, alpha_tex,
                              rgba[:, :, 3].copy() if transparent else None, kind)
    cache[mat_index] = result
    return result


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _cull_transparent(faces, uvs, alpha, threshold):
    """Drop triangles whose texture is see-through everywhere we sample."""
    h, w = alpha.shape
    t = uvs[faces]  # (m,3,2)
    samples = [t[:, 0], t[:, 1], t[:, 2], t.mean(axis=1),
               (t[:, 0] + t[:, 1]) / 2, (t[:, 1] + t[:, 2]) / 2, (t[:, 2] + t[:, 0]) / 2,
               (t[:, 0] * 4 + t[:, 1] + t[:, 2]) / 6, (t[:, 1] * 4 + t[:, 0] + t[:, 2]) / 6,
               (t[:, 2] * 4 + t[:, 0] + t[:, 1]) / 6]
    best = np.zeros(len(faces))
    for s in samples:
        x = np.clip((np.mod(s[:, 0], 1.0) * w).astype(np.int64), 0, w - 1)
        y = np.clip((np.mod(s[:, 1], 1.0) * h).astype(np.int64), 0, h - 1)
        best = np.maximum(best, alpha[y, x])
    return faces[best >= threshold]


def _split(positions, normals, tangents, uvs, ids, weights, faces, limit=MAX_VERTS):
    """Split a mesh into pieces of at most `limit` vertices."""
    if len(positions) <= limit:
        return [(positions, normals, tangents, uvs, ids, weights, faces)]
    pieces = []
    start = 0
    while start < len(faces):
        remap = {}
        end = start
        while end < len(faces):
            new = [v for v in faces[end] if v not in remap]
            if len(remap) + len(set(new)) > limit:
                break
            for v in faces[end]:
                if v not in remap:
                    remap[v] = len(remap)
            end += 1
        order = np.array(sorted(remap, key=remap.get))
        local = np.vectorize(remap.get)(faces[start:end])
        pieces.append((positions[order], normals[order], tangents[order], uvs[order],
                       ids[order], weights[order], local))
        start = end
    return pieces


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def _blank(ref):
    """Same model with its mesh replaced by one invisible triangle."""
    if not ref.bones or not ref.materials:
        return None
    return M.write_mod3(ref, [M.blank_part(ref)], ref.materials, M.BLOCK_SKIN4_1UV, None)


def _sanitize(name):
    s = re.sub(r"[^A-Za-z0-9_]", "_", name)[:40]
    return s or "mat"


@dataclass
class Piece:
    """One armor model that can receive avatar geometry."""
    path: str
    ref: object
    mrl_path: str
    mrl: object
    funcs: set = field(default_factory=set)
    parts: list = field(default_factory=list)
    mat_names: list = field(default_factory=list)
    mat_lookup: dict = field(default_factory=dict)
    chains: list = field(default_factory=list)


def _find_mrl(mod_path, mrls):
    stem = mod_path.rsplit(".", 1)[0].lower()
    exact = next((m for m in mrls if m.rsplit(".", 1)[0].lower() == stem), None)
    if exact:
        return exact
    folder = mod_path.rsplit("\\", 1)[0].lower()
    return next((m for m in mrls if m.rsplit("\\", 1)[0].lower() == folder), None)


def _union_skeleton(pieces):
    pos, parents = {}, {}
    for piece in pieces:
        sk = _hunter_skeleton(piece.ref)
        for f, p in sk.pos.items():
            if f not in pos:
                pos[f] = p
                parents[f] = sk.parent.get(f)
    return HunterSkeleton(pos, parents)


def _nearest_present(f, present, parents):
    seen = set()
    cur = f
    while cur is not None and cur not in seen:
        if cur in present:
            return cur
        seen.add(cur)
        cur = parents.get(cur)
    for fallback in (13, 1, 0):
        if fallback in present:
            return fallback
    return next(iter(present))


# Plain cloth/leather sets whose materials make good templates when the chosen set is
# all metal (their materials are copied, never their textures).
TEMPLATE_SETS = ("pl001_0000", "pl014_0000", "pl053_0000")


def _extra_templates(game, gender, armor):
    out = []
    for other in TEMPLATE_SETS:
        if other == armor:
            continue
        mods, mrls = _armor_files(game, gender, other)
        for m in mrls:
            if _part_of(m) != "body":
                continue
            try:
                mrl = R.read_mrl3(game.read(m))
            except Exception:
                continue
            out += [(mat, mrl) for mat in mrl.materials]
    return out


def _template_score(m):
    """How well a game material suits a VRM surface: a plain, non-metal player material
    that receives the game's wetness and has albedo, normal and RMT slots."""
    roles = {R.slot_role(b.map_type)[0] for b in m.bindings if (b.res_type & 0xF) == R.RES_TEXTURE}
    if "albedo" not in roles:
        return None
    names = R.cbuffer_names(m)
    score = 0
    if "CBMhMaterialPLLocal__disclosure" in names:
        score += 100
    if "CBMhMaterialGlobal" in names:
        score += 30
    if "normal" in roles and "rmt" in roles:
        score += 20
    if R.layout(m) is not None:
        score += 10
    for name, weight in (("fFilmBlend__uiUNorm", 25), ("fMetalic__uiUNorm", 15),
                          ("fTranslucency__uiUNorm", 10), ("fRefractBlend__uiUNorm", 10)):
        v = R.get_field(m, name)
        if isinstance(v, float) and v > 0.05:
            score -= weight
    if "Fur" in " ".join(names) or "Uber" in " ".join(names):
        score -= 40
    return score


def _template_material(piece, name_hash, candidates=()):
    """(material, its mrl3, mesh header) to clone for this piece's VRM materials."""
    ref, mrl = piece.ref, piece.mrl
    idx = _pick_reference_mesh(ref)
    header = ref.meshes[idx] if idx is not None else {}
    pool = [(m, mrl) for m in mrl.materials] + list(candidates)
    best, best_score = None, None
    for m, owner in pool:
        sc = _template_score(m)
        if sc is None:
            continue
        if owner is mrl:
            sc += 1   # prefer the piece's own material on ties
        if best_score is None or sc > best_score:
            best, best_score = (m, owner), sc
    if best is None and mrl.materials:
        best = (mrl.materials[0], mrl)
    if best is None:
        return None, mrl, header
    return best[0], best[1], header


# Material parameters for VRM surfaces (fields that a template doesn't have are skipped).
BASE_PARAMS = {
    "fBaseMapFactor__uiColor": (1.0, 1.0, 1.0, 1.0), "fEmissiveMapFactor__uiColor": (0.0, 0.0, 0.0),
    "fMetalic__uiUNorm": 0.0, "fSpecular__uiUNorm": 0.5, "fTranslucency__uiUNorm": 0.0,
    "bBaseColorEmissive": 0, "bUseCMM": 0, "fFilmBlend__uiUNorm": 0.0, "fRefractBlend__uiUNorm": 0.0,
    "fEmissiveBlendColorBlend__uiUNorm": 0.0, "fDetailNormalBlend__uiUNorm": 0.0, "bUseWaveEmit": 0,
    "fAddColorA__uiColor": (0.0, 0.0, 0.0, 0.0), "fAddColorB__uiColor": (0.0, 0.0, 0.0, 0.0),
    "fAddColorC__uiColor": (0.0, 0.0, 0.0, 0.0), "fAddColorD__uiColor": (0.0, 0.0, 0.0, 0.0),
    "bEmissive": 0,
}
KIND_PARAMS = {
    "skin": {"fSubSurfaceBlend__uiUNorm": 0.5, "fRoughness__uiUNorm": 0.6},
    "hair": {"fSubSurfaceBlend__uiUNorm": 0.0, "fRoughness__uiUNorm": 0.45},
    "eye": {"fSubSurfaceBlend__uiUNorm": 0.0, "fRoughness__uiUNorm": 0.35},
    "cloth": {"fSubSurfaceBlend__uiUNorm": 0.0, "fRoughness__uiUNorm": 0.75},
}


def _hash_function(ref, mrl):
    """Material names are linked by JAMCRC; some tools mask the top bit."""
    hashes = {m.name_hash for m in mrl.materials}
    masked = bool(ref.materials) and all(
        (M.material_hash(n) & 0x7FFFFFFF) in hashes and M.material_hash(n) not in hashes for n in ref.materials)
    return (lambda n: M.material_hash(n) & 0x7FFFFFFF) if masked else M.material_hash


def _write_materials(piece, model, texset, opts, tex_cache, name_hash, candidates=()):
    template, ref_mrl, _ = _template_material(piece, name_hash, candidates)
    if template is None:
        raise ValueError("no usable material found in " + piece.mrl_path)
    new_mrl = R.Mrl3(list(piece.mrl.header), [], [])
    tex_index = {}
    tex_code = ref_mrl.textures[0][0] if ref_mrl.textures else 0x241F5DEB
    tex_unkn = ref_mrl.textures[0][1] if ref_mrl.textures else (0,) * 12

    def tex_slot(path):
        if path not in tex_index:
            new_mrl.textures.append((tex_code, tex_unkn, path))
            tex_index[path] = len(new_mrl.textures)
        return tex_index[path]

    inverse = {v: k for k, v in piece.mat_lookup.items()}
    for local_index, name in enumerate(piece.mat_names):
        texs = _material_textures(model, inverse[local_index], texset, opts, tex_cache)
        m = template.clone()
        m.fields[1] = name_hash(name)
        R.set_fields(m, dict(BASE_PARAMS, **KIND_PARAMS[texs.kind]))
        for b in m.bindings:
            if (b.res_type & 0xF) != R.RES_TEXTURE:
                continue
            role, _slot = R.slot_role(b.map_type)
            path = {"albedo": texs.albedo, "normal": texs.normal, "rmt": texs.rmt,
                    "black": texs.black, "alpha": texs.alpha}.get(role)
            if path is None:
                if 0 < b.tex_idx <= len(ref_mrl.textures):
                    path = ref_mrl.textures[b.tex_idx - 1][2]
                else:
                    b.tex_idx = 0
                    continue
            b.tex_idx = tex_slot(path)
        new_mrl.materials.append(m)
    return R.write_mrl3(new_mrl)


PIECE_ORDER = {"body": 0, "wst": 1, "leg": 2, "arm": 3, "helm": 4, "other": 5}


# ---------------------------------------------------------------------------
# Physics (VRM spring bones -> MHW .ctc chains)
# ---------------------------------------------------------------------------

PHYSICS_IDS = range(150, 247)     # 247-254 are attachment points on the player skeleton
HEAD_BONES = {"head", "leftEye", "rightEye", "jaw"}


@dataclass
class PlannedChain:
    nodes: list            # kept VRM nodes, root first
    funcs: list            # bone function per kept node
    end_func: int
    weight: float          # total vertex weight on the chain (for prioritising)
    radius: float = 0.02   # VRM hit radius (metres, scaled later)


def _node_weights(model):
    w = {}
    for prim in model.primitives:
        for k in range(prim.joints.shape[1]):
            for node, wt in zip(prim.joints[:, k], prim.weights[:, k]):
                if wt > 0:
                    w[int(node)] = w.get(int(node), 0.0) + float(wt)
    return w


def _plan_physics(model, used_funcs, max_new_bones, log):
    """Choose which spring joints become MHW physics bones, within the id budget."""
    free = [f for f in PHYSICS_IDS if f not in used_funcs]
    budget = min(len(free), max_new_bones)
    weights = _node_weights(model)
    chains = []
    for sc in model.springs:
        wsum = sum(weights.get(n, 0.0) for n in sc.nodes)
        if wsum > 0:
            chains.append([list(sc.nodes), wsum, sc.radius])
    if not chains or budget < 2:
        return [], {}

    def cost():
        return sum(len(c[0]) + 1 for c in chains)

    # Thin out long chains first (keep both ends), then drop the least visible chains.
    while cost() > budget:
        longest = max(chains, key=lambda c: len(c[0]))
        if len(longest[0]) > 2:
            keep = longest[0][::2]
            if keep[-1] != longest[0][-1]:
                keep.append(longest[0][-1])
            longest[0] = keep if len(keep) < len(longest[0]) else longest[0][:-1]
        else:
            chains.remove(min(chains, key=lambda c: c[1]))
            if not chains:
                break
    ids = iter(free)
    planned, node_funcs = [], {}
    total = sum(len(sc.nodes) for sc in model.springs)
    for kept, wsum, radius in chains:
        funcs = [next(ids) for _ in kept]
        planned.append(PlannedChain(kept, funcs, next(ids), wsum, radius))
        node_funcs.update(zip(kept, funcs))
    # Joints that were thinned out follow the nearest kept joint above them.
    for sc in model.springs:
        last = None
        for n in sc.nodes:
            if n in node_funcs:
                last = node_funcs[n]
            elif last is not None:
                node_funcs[n] = last
    kept_total = sum(len(c.nodes) for c in planned)
    log("  physics: %d chains, %d of %d spring joints simulated" % (len(planned), kept_total, total))
    return planned, node_funcs


AUTO_CAPSULES = ((1, 2), (2, 3), (6, 7), (10, 11), (7, 8), (11, 12), (14, 15), (18, 19), (15, 16), (19, 20))


def _seg_distance(p, a, b):
    ab = b - a
    t = np.clip((p - a) @ ab / max(float(ab @ ab), 1e-12), 0, 1)
    return np.linalg.norm(p - (a + np.outer(t, ab)), axis=1)


def _capsules(model, fit, ref, log):
    """Collision capsules for the avatar's physics, on base bones of `ref`."""
    worlds = ref.bone_world_matrices()
    index = ref.function_to_index()

    def local(f, p):
        return (np.linalg.inv(worlds[index[f]]) @ np.append(p, 1.0))[:3]

    caps = []
    if model.colliders:
        for c in model.colliders:
            f = fit.bone_function.get(fit.node_bone.get(c.node))
            if f not in index:
                continue
            w = model.world[c.node]
            p0 = fit.map_point(c.node, (w @ np.append(c.offset, 1.0))[:3])
            p1 = fit.map_point(c.node, (w @ np.append(c.tail, 1.0))[:3]) if c.tail is not None else p0
            r = c.radius * fit.scale
            caps.append((f, local(f, p0), r, f, local(f, p1), r))
        log("  collision: %d colliders from the VRM" % len(caps))
        return caps
    # No colliders in the VRM: fit capsules around the avatar's own body.
    pts, fns = [], []
    for pi, prim in enumerate(model.primitives):
        mat = model.materials[prim.material] if 0 <= prim.material < len(model.materials) else None
        if mat is not None and material_kind(mat.name) == "hair":
            continue
        strong = fit.weights[pi][:, 0] > 0.6
        pts.append(fit.positions[pi][strong])
        fns.append(fit.functions[pi][strong, 0])
    if not pts:
        return caps
    pts, fns = np.vstack(pts), np.concatenate(fns)
    for fa, fb in AUTO_CAPSULES:
        if fa not in index or fb not in index:
            continue
        sel = pts[fns == fa]
        if len(sel) < 8:
            continue
        a, b = worlds[index[fa]][:3, 3], worlds[index[fb]][:3, 3]
        r = float(np.median(_seg_distance(sel, a, b))) * 0.95
        if r > 0:
            caps.append((fa, local(fa, a), r, fb, local(fb, b), r))
    head = pts[fns == 4]
    if 4 in index and len(head) >= 8:
        lo, hi = head.min(axis=0), head.max(axis=0)
        c = (lo + hi) / 2
        r = 0.45 * float(np.min(hi - lo))
        caps.append((4, local(4, c), r, 4, local(4, c), r))
    log("  collision: %d capsules fitted to the avatar's body" % len(caps))
    return caps


def _hunter_neck_points(game, gender, skeleton_ref):
    """The hunter's own face/neck model posed on our (moved) skeleton: its neck points."""
    mods = [p for p in game.listdir("pl\\%s_face\\face000\\" % gender) if p.lower().endswith(".mod3")]
    best = None
    for p in mods:
        try:
            ref = M.read_mod3(game.read(p))
        except Exception:
            continue
        if best is None or ref.header["vertexCount"] > best.header["vertexCount"]:
            best = ref
    if best is None or not best.bones:
        return None
    face_w = best.bone_world_matrices()
    ours = skeleton_ref.bone_world_matrices()
    our_index = skeleton_ref.function_to_index()
    # Per face-model bone: matrix taking it from the hunter's rest pose onto our skeleton.
    moves = []
    for i, b in enumerate(best.bones):
        j = our_index.get(b.function)
        moves.append(np.eye(4) if j is None else ours[j] @ np.linalg.inv(face_w[i]))
    moves = np.array(moves)
    out = []
    for m in range(len(best.meshes)):
        skin = best.mesh_skin(m)
        if skin is None:
            continue
        pos, ids, w = skin
        ids = np.clip(ids, 0, len(moves) - 1)
        homo = np.hstack([pos, np.ones((len(pos), 1))])
        posed = np.einsum("nk,nkij,nj->ni", w, moves[ids], homo)[:, :3]
        main = np.array([best.bones[i].function for i in ids[np.arange(len(ids)), w.argmax(axis=1)]])
        out.append(posed[main == 3])   # the neck itself, not the face or the chest
    pts = np.vstack(out) if out else np.zeros((0, 3))
    return pts if len(pts) else None


def _cover_neck(model, fit, neck_pts, skeleton_ref, margin=0.4, max_growth=1.6):
    """Push the avatar's neck surface out to just cover the hunter's neck. Returns moved count."""
    index = skeleton_ref.function_to_index()
    if neck_pts is None or 3 not in index or 4 not in index:
        return 0
    worlds = skeleton_ref.bone_world_matrices()
    a, b = worlds[index[3]][:3, 3], worlds[index[4]][:3, 3]
    axis = b - a
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return 0
    axis /= length
    ref = np.cross(axis, [1.0, 0.0, 0.0])
    if np.linalg.norm(ref) < 0.1:
        ref = np.cross(axis, [0.0, 1.0, 0.0])
    ref /= np.linalg.norm(ref)
    ref2 = np.cross(axis, ref)

    def cyl(p):
        d = p - a
        h = d @ axis
        radial = d - np.outer(h, axis)
        r = np.linalg.norm(radial, axis=1)
        ang = np.arctan2(radial @ ref2, radial @ ref)
        return h, r, ang, radial

    nb_h, nb_a = 12, 24
    hh, rr, aa, _ = cyl(neck_pts)
    # Work in the height range the hunter's neck actually spans, near our neck bone.
    h0 = max(float(hh.min()), -1.0 * length) - 1e-6
    h1 = min(float(hh.max()), 1.2 * length) + 1e-6
    if h1 <= h0:
        return 0
    keep = (hh >= h0) & (hh <= h1)
    if not np.any(keep):
        return 0
    hb = np.clip(((hh[keep] - h0) / (h1 - h0) * nb_h).astype(int), 0, nb_h - 1)
    ab = ((aa[keep] + np.pi) / (2 * np.pi) * nb_a).astype(int) % nb_a
    need = np.zeros((nb_h, nb_a))
    np.maximum.at(need, (hb, ab), rr[keep])
    # Fill gaps (coarse meshes leave empty cells): around each ring, then between rings.
    cols = np.arange(nb_a)
    for i in range(nb_h):
        have = need[i] > 0
        if have.any() and not have.all():
            need[i] = np.interp(cols, cols[have], need[i][have], period=nb_a)
    rows = np.arange(nb_h)
    filled = need.max(axis=1) > 0
    if filled.sum() >= 2:
        lo, hi = rows[filled].min(), rows[filled].max()
        for j in range(nb_a):
            inner = (rows >= lo) & (rows <= hi)
            need[inner, j] = np.interp(rows[inner], rows[filled], need[filled, j])
    # Smooth: each cell needs the most of itself and its neighbours.
    grown = need.copy()
    for dh in (-1, 0, 1):
        for da in (-1, 0, 1):
            grown = np.maximum(grown, np.roll(np.roll(need, dh, axis=0), da, axis=1) * (1 if dh == 0 else 0.97))
    typical = float(np.median(rr[keep]))
    moved = 0
    for pi, prim in enumerate(model.primitives):
        mat = model.materials[prim.material] if 0 <= prim.material < len(model.materials) else None
        if mat is not None and material_kind(mat.name) == "hair":
            continue
        pos = fit.positions[pi]
        main = fit.functions[pi][:, 0]
        cand = main == 3
        if not np.any(cand):
            continue
        h, r, ang, radial = cyl(pos)
        inside = cand & (h >= h0) & (h <= h1) & (r < 2.5 * typical) & (r > 1e-6)
        if not np.any(inside):
            continue
        hbin = np.clip(((h - h0) / (h1 - h0) * nb_h).astype(int), 0, nb_h - 1)
        abin = ((ang + np.pi) / (2 * np.pi) * nb_a).astype(int) % nb_a
        # Never more than max_growth times thicker, so the neck keeps its character.
        target = np.minimum(grown[hbin, abin] + margin, r * max_growth)
        push = inside & (grown[hbin, abin] > 0) & (r < target)
        if np.any(push):
            scale = (target[push] / r[push])[:, None]
            pos[push] = a + np.outer(h[push], axis) + radial[push] * scale
            moved += int(push.sum())
    return moved


def _ctc_template(game, mods):
    """Tuning values from a real .ctc: prefer hair, then this armor set, then defaults."""
    candidates = [p for p in game.listdir("pl\\hair\\") if p.lower().endswith(".ctc")]
    for m in mods:
        folder = m.rsplit("\\", 1)[0]
        candidates += [p for p in game.listdir(folder + "\\") if p.lower().endswith(".ctc")]
    for path in candidates[:40]:
        try:
            t = C.parse_template(game.read(path))
        except Exception:
            t = None
        if t is not None:
            return t, path
    return C.default_template(), None


def _sibling(game, mod_path, ext):
    """Existing file with the same stem (or any file of that type) next to a model."""
    folder, name = mod_path.rsplit("\\", 1)
    stem = name.rsplit(".", 1)[0].lower()
    files = [p for p in game.listdir(folder + "\\") if p.lower().endswith(ext)]
    exact = [p for p in files if p.rsplit("\\", 1)[1].rsplit(".", 1)[0].lower() == stem]
    return (exact or files or [None])[0]


# ---------------------------------------------------------------------------
# Helmet coverage (does the helmet hide the hunter's face?)
# ---------------------------------------------------------------------------

def _model_points(ref):
    pts = []
    for i in range(len(ref.meshes)):
        geo = ref.mesh_geometry(i)
        if geo is not None:
            pts.append(geo[0])
    return np.vstack(pts) if pts else np.zeros((0, 3))


def _model_triangles(ref):
    tris = []
    for i in range(len(ref.meshes)):
        geo = ref.mesh_geometry(i)
        if geo is not None and len(geo[2]):
            tris.append(geo[0][geo[2]])
    return np.concatenate(tris) if tris else np.zeros((0, 3, 3))


def _ray_hits(origin, direction, tris):
    """Does a ray hit any triangle (Moller-Trumbore, both sides)?"""
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    pvec = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tvec = origin - tris[:, 0]
    u = np.einsum("ij,ij->i", tvec, pvec) * inv
    qvec = np.cross(tvec, e1)
    v = (qvec @ direction) * inv
    t = np.einsum("ij,ij->i", e2, qvec) * inv
    return bool(np.any(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)))


def _face_points(game, gender):
    mods = [p for p in game.listdir("pl\\%s_face\\face000\\" % gender) if p.lower().endswith(".mod3")]
    best = None
    for p in mods:
        try:
            pts = _model_points(M.read_mod3(game.read(p)))
        except Exception:
            continue
        if best is None or len(pts) > len(best):
            best = pts
    return best


def helm_coverage(helm_ref, face_pts):
    """Fraction of the hunter's face (front-most points) covered by the helmet mesh, 0..1."""
    if face_pts is None or len(face_pts) == 0 or not helm_ref.bones:
        return None
    sk = _hunter_skeleton(helm_ref)
    need = (4, 16, 17, 20, 21)
    if not all(sk.has(f) for f in need):
        return None
    up = sk.p(4) - (sk.p(16) + sk.p(20)) / 2
    height = np.linalg.norm(up)
    up /= height
    fwd = (sk.p(17) - sk.p(16) + sk.p(21) - sk.p(20)) / 2
    fwd -= up * np.dot(fwd, up)
    if np.linalg.norm(fwd) < 1e-6:
        return None
    fwd /= np.linalg.norm(fwd)
    tris = _model_triangles(helm_ref)
    if len(tris) == 0:
        return 0.0
    depth = (face_pts - sk.p(4)) @ fwd
    front = face_pts[depth >= np.percentile(depth, 85)]
    front = front[:: max(1, len(front) // 200)]
    # A face point is covered if a ray going forward out of the face hits the helmet.
    covered = sum(_ray_hits(p, fwd, tris) for p in front)
    return covered / len(front)


def scan_armor(game, gender="f", log=print):
    """[(armor id, coverage or None)] for every armor set with a helmet."""
    face = _face_points(game, gender)
    out = []
    sets = armor_sets(game)
    for i, armor in enumerate(sets):
        mods, _ = _armor_files(game, gender, armor)
        helms = [p for p in mods if _part_of(p) == "helm"]
        cov = None
        if helms:
            try:
                ref = M.read_mod3(game.read(max(helms, key=len)))
                cov = helm_coverage(ref, face)
            except Exception:
                cov = None
        out.append((armor, cov))
        if i % 25 == 0:
            log("Checked %d of %d armor sets" % (i + 1, len(sets)))
    return out


FULL_HELMET = 0.8


def build(vrm_path, game, opts=None, log=print):
    opts = opts or Options()
    log("Loading %s" % vrm_path)
    model = load_vrm(vrm_path)
    log("VRM %s.0 model: %d mesh pieces, %d materials, %d spring bone chains" % (
        "1" if model.version == 1 else "0", len(model.primitives), len(model.materials), len(model.springs)))
    if not model.primitives:
        raise ValueError("the VRM has no meshes")
    result = BuildResult()
    armor = opts.armor.lower()
    texset = None
    tex_cache = {}
    template = None

    for gender in opts.genders:
        label = "female" if gender == "f" else "male"
        mods, mrls = _armor_files(game, gender, armor)
        if not mods:
            log("No %s armor files for %s, skipping" % (label, armor))
            continue
        refs = {p: M.read_mod3(game.read(p)) for p in mods}
        pieces = []
        for path in mods:
            mrl_path = _find_mrl(path, mrls)
            if refs[path].bones and mrl_path:
                pieces.append(Piece(path, refs[path], mrl_path, R.read_mrl3(game.read(mrl_path))))
        pieces.sort(key=lambda pc: (PIECE_ORDER.get(_part_of(pc.path), 5), -pc.ref.header["vertexCount"]))
        if not pieces or _part_of(pieces[0].path) != "body":
            raise ValueError("armor %s has no usable body model for %s hunters" % (armor, label))
        body = pieces[0]
        helm = next((pc for pc in pieces if _part_of(pc.path) == "helm"), None)
        log("Reference: %s (%d bones, %d meshes)" % (body.path, len(body.ref.bones), len(body.ref.meshes)))
        if helm is None:
            log("  this set has no helmet model; the whole avatar goes in the body model")
        elif gender == opts.genders[0]:
            cov = helm_coverage(helm.ref, _face_points(game, gender))
            if cov is not None and cov < FULL_HELMET:
                log("  NOTE: this set's helmet is open-faced, so your hunter's own face may show "
                    "through the avatar's head. A set marked 'full helmet' avoids that.")

        # Physics bones: plan ids that no piece of this set already uses.
        used = set()
        for pc in pieces:
            used |= {b.function for b in pc.ref.bones}
        planned, node_funcs = [], {}
        if opts.physics and model.springs:
            room = 255 - max(len(pc.ref.bones) for pc in pieces)
            planned, node_funcs = _plan_physics(model, used, room, log)

        sk = _union_skeleton(pieces)
        fit = fit_vrm(model, sk, opts.height_scale, node_funcs, opts.keep_proportions)
        for n in fit.notes:
            log("  " + n)
        if opts.keep_proportions:
            # Move the hunter's joints to the avatar's joints so its proportions are kept.
            for pc in pieces:
                pc.ref = M.repose_skeleton(pc.ref, fit.joint_targets)
            refs = {p: M.repose_skeleton(r, fit.joint_targets) for p, r in refs.items()}
            log("  kept the avatar's own proportions (%d joints moved)" % len(fit.joint_targets))

        # Add the physics bones to the piece each chain hangs from (head -> helmet).
        piece_chains = {}
        for chain in planned:
            root = chain.nodes[0]
            target = helm if (helm is not None and fit.node_bone.get(root) in HEAD_BONES) else body
            parent = model.parents[root]
            anchor = node_funcs.get(parent) if parent in node_funcs else \
                fit.bone_function.get(fit.node_bone.get(parent if parent >= 0 else root), 0)
            piece_chains.setdefault(id(target), (target, []))[1].append((chain, anchor))
        for target, chains in piece_chains.values():
            new = []
            have = {b.function for b in target.ref.bones}
            for chain, anchor in chains:
                parent = anchor if anchor in have else _nearest_present(anchor, have, sk.parent)
                pos = [fit.node_positions[n] for n in chain.nodes]
                for f, p in zip(chain.funcs, pos):
                    new.append((f, parent, p))
                    sk.parent[f] = parent
                    parent = f
                if len(pos) >= 2:
                    tip = pos[-1] + (pos[-1] - pos[-2])
                else:
                    base = target.ref.bone_world_matrices()[target.ref.function_to_index()[new[-1][1]]][:3, 3] \
                        if new[-1][1] in have else pos[-1]
                    step = pos[-1] - base
                    if np.linalg.norm(step) < 1e-6:
                        step = np.array([0.0, 0.0, -1.0]) * fit.scale * 0.05
                    tip = pos[-1] + step * 0.5
                new.append((chain.end_func, parent, tip))
                sk.parent[chain.end_func] = parent
                pts = pos + [tip]
                dirs = [pts[i + 1] - pts[i] for i in range(len(pts) - 1)]
                dirs.append(dirs[-1])
                target.chains.append({"funcs": [*chain.funcs, chain.end_func], "dirs": dirs,
                                      "radius": max(chain.radius * fit.scale, 0.3)})
            target.ref = M.extend_skeleton(target.ref, new)
        for pc in pieces:
            pc.funcs = {b.function for b in pc.ref.bones}

        if texset is None:
            with_ext = any(t[2].lower().endswith(".tex") for t in body.mrl.textures)
            tex_dir = body.path.rsplit("\\", 1)[0]
            if tex_dir.lower().endswith("\\mod"):
                tex_dir = tex_dir[:-4] + "\\tex"
            texset = TextureSet(tex_dir + "\\vroidhunter", with_ext)

        if opts.cover_neck:
            neck = _hunter_neck_points(game, gender, body.ref)
            moved = _cover_neck(model, fit, neck, body.ref)
            if moved:
                log("  thickened the neck slightly (%d vertices) so your hunter's own neck stays hidden" % moved)

        head_funcs = {f for f in (fit.bone_function.get(b) for b in HEAD_BONES) if f is not None}
        if helm is not None:
            head_funcs |= {f for c in helm.chains for f in c["funcs"]}
        candidates = [(m, pc.mrl) for pc in pieces for m in pc.mrl.materials]
        candidates += _extra_templates(game, gender, armor)
        ref_sign = _reference_winding(body.ref)
        culled = 0
        for pi, prim in enumerate(model.primitives):
            mi = prim.material
            mat = model.materials[mi] if 0 <= mi < len(model.materials) else None
            textures = _material_textures(model, mi, texset, opts, tex_cache)
            faces = prim.faces
            if opts.alpha_cull and textures.alpha_pixels is not None:
                thresh = 255 * (mat.alpha_cutoff if mat.alpha_mode == "MASK" else 0.05)
                before = len(faces)
                faces = _cull_transparent(faces, prim.uvs, textures.alpha_pixels, max(1.0, thresh))
                culled += before - len(faces)
            if len(faces) == 0:
                continue
            pos, nrm, tan = fit.positions[pi], fit.normals[pi], fit.tangents[pi]
            if tan is None:
                tan = compute_tangents(pos, nrm, prim.uvs, faces)
            funcs, wts, uvs = fit.functions[pi], fit.weights[pi], prim.uvs
            if _winding_sign(pos, nrm, faces) * ref_sign < 0:
                faces = faces[:, [0, 2, 1]]

            active = wts > 0
            owner = np.full(len(faces), -1)
            # The head (and hair) goes in the helmet model, like the game's own helmets.
            if helm is not None:
                in_head = np.vectorize(lambda f: int(f) in head_funcs)(funcs) | ~active
                head_ok = in_head.all(axis=1)[faces].all(axis=1)
                helm_ok = (np.vectorize(lambda f: int(f) in helm.funcs)(funcs) | ~active).all(axis=1)
                owner[head_ok & helm_ok[faces].all(axis=1)] = pieces.index(helm)
            # Everything else: the first piece whose skeleton has all its bones (body first).
            for k, pc in enumerate(pieces):
                if pc is helm:
                    continue
                has = np.vectorize(lambda f: int(f) in pc.funcs)(funcs) | ~active
                face_ok = has.all(axis=1)[faces].all(axis=1) & (owner < 0)
                owner[face_ok] = k
            owner[owner < 0] = 0

            for k in np.unique(owner):
                pc = pieces[k]
                sub = faces[owner == k]
                used_v = np.unique(sub)
                remap = -np.ones(len(pos), dtype=np.int64)
                remap[used_v] = np.arange(len(used_v))
                f2i = pc.ref.function_to_index()
                ids = np.vectorize(lambda f: f2i[_nearest_present(int(f), f2i, sk.parent)])(funcs[used_v])
                p_, n_, t_, u_, w_ = pos[used_v], nrm[used_v], tan[used_v], uvs[used_v], wts[used_v]
                sub = remap[sub]
                if opts.double_sided and mat is not None and mat.double_sided:
                    n = len(p_)
                    back_t = t_.copy()
                    back_t[:, 3] *= -1
                    p_, n_, t_ = np.vstack([p_, p_]), np.vstack([n_, -n_]), np.vstack([t_, back_t])
                    u_, ids, w_ = np.vstack([u_, u_]), np.vstack([ids, ids]), np.vstack([w_, w_])
                    sub = np.vstack([sub, sub[:, [0, 2, 1]] + n])
                if mi not in pc.mat_lookup:
                    pc.mat_lookup[mi] = len(pc.mat_names)
                    pc.mat_names.append("VRH_%02d_%s" % (len(pc.mat_names), _sanitize(mat.name if mat else "default")))
                for piece in _split(p_, n_, t_, u_, ids, w_, sub):
                    pp, nn, tt, uu, ii, ww, ff = piece
                    pc.parts.append(M.MeshPart(pc.mat_lookup[mi], pp, nn, tt, uu, ii, ww, ff))
        if culled:
            log("  removed %d fully transparent triangles" % culled)
        if not any(pc.parts for pc in pieces):
            raise ValueError("nothing left to draw after processing the VRM")

        for pc in pieces:
            if not pc.parts:
                continue
            name_hash = _hash_function(pc.ref, pc.mrl)
            _, _, header = _template_material(pc, name_hash)
            blocktype = M.BLOCK_SKIN4_1UV
            if header and M.BLOCK_TYPES[header["blocktype"]]["weights"] == 8:
                blocktype = M.BLOCK_SKIN8_1UV
            result.files[pc.mrl_path] = _write_materials(pc, model, texset, opts, tex_cache, name_hash,
                                                         candidates)
            result.files[pc.path] = M.write_mod3(pc.ref, pc.parts, pc.mat_names, blocktype, header)
            log("  %s: %d mesh parts, %d vertices" % (pc.path.rsplit("\\", 1)[-1], len(pc.parts),
                                                     sum(len(p.positions) for p in pc.parts)))
            if pc.chains:
                if template is None:
                    template, source = _ctc_template(game, mods)
                    log("  physics tuning copied from %s" % (source or "built-in defaults"))
                stem = pc.path.rsplit(".", 1)[0]
                # Our own colliders replace the armor's: its capsules fit the armor, not the avatar.
                capsules = _capsules(model, fit, pc.ref, log if pc is body else (lambda m: None))
                ccl_src = _sibling(game, pc.path, ".ccl") or _sibling(game, body.path, ".ccl")
                unknowns = CL.template_unknowns(game.read(ccl_src)) if ccl_src else ((0,) * 8, (0,) * 4)
                ctc_path = _sibling(game, pc.path, ".ctc") or stem + ".ctc"
                ccl_path = _sibling(game, pc.path, ".ccl") or stem + ".ccl"
                result.files[ctc_path] = C.write_ctc(template, pc.chains, collide=bool(capsules))
                result.files[ccl_path] = CL.write_ccl(capsules, unknowns)
                log("  %s: %d physics chains, %d collision capsules" % (
                    ctc_path.rsplit("\\", 1)[-1], len(pc.chains), len(capsules)))

        # Armor pieces the avatar doesn't use draw nothing, so they don't poke through.
        hidden = 0
        for path in mods:
            if path in result.files:
                continue
            blank = _blank(refs[path])
            if blank is not None:
                result.files[path] = blank
                hidden += 1
        if hidden:
            log("  %d unused %s armor pieces made invisible" % (hidden, label))

    if not result.files:
        raise ValueError("armor %s was not found in the game files" % armor)
    if texset:
        result.files.update(texset.files)
    result.notes.append("%d files generated" % len(result.files))
    return result
