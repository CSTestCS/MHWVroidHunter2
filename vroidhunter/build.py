"""Turn a VRM file into the nativePC files that replace one armor set."""

import io
import re
from dataclasses import dataclass, field

import numpy as np

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
    hide_face_and_hair: bool = True
    height_scale: float = 1.0
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
    rmt = texset.solid("vrh_matte_RMT", (220, 0, 0, 0))
    black = texset.solid("vrh_black", (0, 0, 0, 255))
    alpha_tex = texset.solid("vrh_white", (255, 255, 255, 255))
    if transparent:
        a = rgba[:, :, 3]
        alpha_tex = texset.add(tag + "_ALP", np.dstack([a, a, a, np.full_like(a, 255)]), X.FMT_RGBA8)
    result = MaterialTextures(albedo, normal, rmt, black, alpha_tex,
                              rgba[:, :, 3].copy() if transparent else None)
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


def _template_material(piece, name_hash):
    ref, mrl = piece.ref, piece.mrl
    idx = _pick_reference_mesh(ref)
    header = ref.meshes[idx] if idx is not None else {}
    template = None
    if header and header["materialIdx"] < len(ref.materials):
        want = name_hash(ref.materials[header["materialIdx"]])
        template = next((m for m in mrl.materials if m.name_hash == want), None)
    if template is None:
        template = next((m for m in mrl.materials
                         if any(R.slot_role(b.map_type)[0] == "albedo" for b in m.bindings)), None)
    if template is None and mrl.materials:
        template = mrl.materials[0]
    return template, header


def _hash_function(ref, mrl):
    """Material names are linked by JAMCRC; some tools mask the top bit."""
    hashes = {m.name_hash for m in mrl.materials}
    masked = bool(ref.materials) and all(
        (M.material_hash(n) & 0x7FFFFFFF) in hashes and M.material_hash(n) not in hashes for n in ref.materials)
    return (lambda n: M.material_hash(n) & 0x7FFFFFFF) if masked else M.material_hash


def _write_materials(piece, model, texset, opts, tex_cache, name_hash):
    template, _ = _template_material(piece, name_hash)
    if template is None:
        raise ValueError("no usable material found in " + piece.mrl_path)
    ref_mrl = piece.mrl
    new_mrl = R.Mrl3(list(ref_mrl.header), [], [])
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


def build(vrm_path, game, opts=None, log=print):
    opts = opts or Options()
    log("Loading %s" % vrm_path)
    model = load_vrm(vrm_path)
    log("VRM %s.0 model with %d mesh pieces and %d materials" % (
        "1" if model.version == 1 else "0", len(model.primitives), len(model.materials)))
    if not model.primitives:
        raise ValueError("the VRM has no meshes")
    result = BuildResult()
    armor = opts.armor.lower()
    texset = None
    tex_cache = {}

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
        for pc in pieces:
            pc.funcs = {b.function for b in pc.ref.bones}
        log("Reference: %s (%d bones, %d meshes)" % (body.path, len(body.ref.bones), len(body.ref.meshes)))

        # Fit the avatar to the hunter skeleton (all pieces' bones combined).
        sk = _union_skeleton(pieces)
        fit = fit_vrm(model, sk, opts.height_scale)
        for n in fit.notes:
            log("  " + n)

        if texset is None:
            with_ext = any(t[2].lower().endswith(".tex") for t in body.mrl.textures)
            tex_dir = body.path.rsplit("\\", 1)[0]
            if tex_dir.lower().endswith("\\mod"):
                tex_dir = tex_dir[:-4] + "\\tex"
            texset = TextureSet(tex_dir + "\\vroidhunter", with_ext)

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

            # Give each triangle to the first piece whose skeleton has all its bones.
            active = wts > 0
            owner = np.full(len(faces), -1)
            for k, pc in enumerate(pieces):
                has = np.vectorize(lambda f: int(f) in pc.funcs)(funcs) | ~active
                vert_ok = has.all(axis=1)
                face_ok = vert_ok[faces].all(axis=1) & (owner < 0)
                owner[face_ok] = k
            owner[owner < 0] = 0

            for k in np.unique(owner):
                pc = pieces[k]
                sub = faces[owner == k]
                used = np.unique(sub)
                remap = -np.ones(len(pos), dtype=np.int64)
                remap[used] = np.arange(len(used))
                f2i = pc.ref.function_to_index()
                ids = np.vectorize(lambda f: f2i[_nearest_present(int(f), f2i, sk.parent)])(funcs[used])
                p_, n_, t_, u_, w_ = pos[used], nrm[used], tan[used], uvs[used], wts[used]
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
            _, header = _template_material(pc, name_hash)
            blocktype = M.BLOCK_SKIN4_1UV
            if header and M.BLOCK_TYPES[header["blocktype"]]["weights"] == 8:
                blocktype = M.BLOCK_SKIN8_1UV
            result.files[pc.mrl_path] = _write_materials(pc, model, texset, opts, tex_cache, name_hash)
            result.files[pc.path] = M.write_mod3(pc.ref, pc.parts, pc.mat_names, blocktype, header)
            log("  %s: %d mesh parts, %d vertices" % (pc.path.rsplit("\\", 1)[-1], len(pc.parts),
                                                     sum(len(p.positions) for p in pc.parts)))

        # Every other model in the set draws nothing.
        hidden = 0
        for path in mods:
            if path in result.files:
                continue
            blank = _blank(refs[path])
            if blank is not None:
                result.files[path] = blank
                hidden += 1
        log("  hid %d other %s armor models" % (hidden, label))

        if opts.hide_face_and_hair:
            count = 0
            for folder in ("pl\\%s_face\\" % gender, "pl\\hair\\"):
                for path in game.listdir(folder):
                    if not path.lower().endswith(".mod3") or path in result.files:
                        continue
                    try:
                        blank = _blank(M.read_mod3(game.read(path)))
                    except Exception:
                        blank = None
                    if blank is not None:
                        result.files[path] = blank
                        count += 1
            log("  hid %d face and hair models" % count)

    if not result.files:
        raise ValueError("armor %s was not found in the game files" % armor)
    if texset:
        result.files.update(texset.files)
    result.notes.append("%d files generated" % len(result.files))
    return result
