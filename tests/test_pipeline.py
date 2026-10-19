import struct

import numpy as np
import pytest

from vroidhunter import mod3 as M
from vroidhunter import mrl3 as R
from vroidhunter import tex as X
from vroidhunter.binio import jamcrc
from vroidhunter.build import Options, armor_sets, build
from vroidhunter.chunk import FolderFiles
from vroidhunter.vrm import load_vrm_bytes

from synth import hunter_bones, make_game_folder, make_vrm


def decode_vertices(mod, i):
    h = mod.meshes[i]
    stride = h["blockSize"]
    start = mod.header["vertexOffset"] + h["vertexOffset"] + stride * (h["vertexSub"] + h["vertexBase"])
    raw = np.frombuffer(mod.data[start:start + stride * h["vertexCount"]], np.uint8).reshape(-1, stride)
    pos = raw[:, 0:12].copy().view("<f4").reshape(-1, 3)
    nrm = raw[:, 12:15].copy().view(np.int8).astype(float)
    packed = raw[:, 24:28].copy().view("<u4").ravel()
    w = np.stack([(packed >> (10 * k)) & 1023 for k in range(3)], axis=1) / 1023.0
    w = np.hstack([w, 1 - w.sum(axis=1, keepdims=True)])
    ids = raw[:, 28:32]
    fstart = mod.header["facesOffset"] + h["faceOffset"] * 2
    faces = np.frombuffer(mod.data[fstart:fstart + 2 * h["faceCount"]], "<u2").reshape(-1, 3).astype(int)
    return pos, nrm, w, ids, faces - h["vertexSub"]


@pytest.fixture(params=[(0, 1, False), (1, 1, False), (0, -1, True)], ids=["vrm0", "vrm1", "vrm0-back-flipped"])
def built(request, tmp_path):
    version, facing, flip = request.param
    data, vrm_joints = make_vrm(version, facing)
    vrm_path = tmp_path / "avatar.vrm"
    vrm_path.write_bytes(data)
    game_root = tmp_path / "game"
    make_game_folder(str(game_root), flip_winding=flip)
    files = FolderFiles(str(game_root))
    result = build(str(vrm_path), files, Options(), log=lambda m: None)
    return result, files, flip


def test_vrm_loader_reads_humanoid():
    data, joints = make_vrm(0)
    model = load_vrm_bytes(data)
    assert model.version == 0
    assert np.allclose(model.bone_position("leftHand"), joints["leftHand"])
    assert len(model.primitives) == 2
    assert model.materials[1].alpha_mode == "MASK"
    model1 = load_vrm_bytes(make_vrm(1)[0])
    assert model1.version == 1


def test_expected_files(built):
    result, files, _ = built
    paths = set(result.files)
    for g in ("f", "m"):
        for part in ("helm", "body", "arm", "wst", "leg"):
            assert "pl\\%s_equip\\pl001_0000\\%s\\mod\\%s_%s001.mod3" % (g, part, g, part) in paths
        assert "pl\\%s_equip\\pl001_0000\\body\\mod\\%s_body001.mrl3" % (g, g) in paths
        assert "pl\\%s_face\\face000\\mod\\face000.mod3" % g in paths
    assert "pl\\hair\\hair000\\mod\\hair000.mod3" in paths
    texes = [p for p in paths if p.endswith(".tex")]
    assert texes, "no textures written"
    for p in texes:
        mips, w, h, fmt, offsets = X.read_tex_header(result.files[p])
        assert w >= 4 and h >= 4 and mips >= 1
        assert offsets[0] == 0xB8 + 8 * mips
        assert list(offsets) == sorted(offsets)
        assert offsets[-1] < len(result.files[p])


def test_body_mesh_fits_hunter(built):
    result, files, flip = built
    body = M.read_mod3(result.files["pl\\f_equip\\pl001_0000\\body\\mod\\f_body001.mod3"])
    ref = M.read_mod3(files.read("pl\\f_equip\\pl001_0000\\body\\mod\\f_body001.mod3"))
    assert body.skeleton_raw == ref.skeleton_raw
    assert body.header["meshCount"] >= 2
    func_of = {i: b.function for i, b in enumerate(body.bones)}
    bones = hunter_bones()
    all_pos, all_funcs, body_z = [], [], []
    for i in range(body.header["meshCount"]):
        pos, nrm, w, ids, faces = decode_vertices(body, i)
        assert faces.min() >= 0 and faces.max() < len(pos)
        assert np.all(w >= -1e-3) and np.allclose(w.sum(axis=1), 1, atol=1e-6)
        assert ids.max() < len(body.bones)
        all_pos.append(pos)
        if "Hair" not in body.materials[body.meshes[i]["materialIdx"]]:
            body_z.append(pos[:, 2])
        all_funcs.append(np.vectorize(func_of.get)(ids[:, 0]))
        # Winding must agree with the reference model's convention.
        fn = np.cross(pos[faces[:, 1]] - pos[faces[:, 0]], pos[faces[:, 2]] - pos[faces[:, 0]])
        vn = nrm[faces].sum(axis=1)
        agree = np.sum(fn * vn, axis=1)
        frac = np.mean(agree > 0) if not flip else np.mean(agree < 0)
        assert frac > 0.9
    pos = np.vstack(all_pos)
    funcs = np.concatenate(all_funcs)
    # Height: ankle-to-head span matches the hunter (Z up in the fake game).
    z = np.concatenate(body_z)
    assert z.max() == pytest.approx(156, abs=6)
    assert z.min() == pytest.approx(9 - 6, abs=8)
    # Vertices driven by each hunter bone sit near that bone.
    for f in (3, 7, 8, 15, 19):
        target = np.array(bones[f][1])
        pts = pos[funcs == f]
        assert len(pts), "no vertices weighted to bone %d" % f
        d = np.linalg.norm(pts - target, axis=1).min()
        assert d < 10, (f, d)
    # Left hand vertices must be on the hunter's left side (-X in the fake game).
    assert np.mean(pos[funcs == 8][:, 0]) < 0
    assert np.mean(pos[funcs == 12][:, 0]) > 0
    # Toes point the way the hunter faces (-Y).
    assert np.mean(pos[funcs == 16][:, 1]) < -2


def test_transparent_triangles_culled(built):
    result, _, _ = built
    body = M.read_mod3(result.files["pl\\f_equip\\pl001_0000\\body\\mod\\f_body001.mod3"])
    hair_mats = [i for i, n in enumerate(body.materials) if "Hair" in n]
    assert hair_mats
    hair_faces = sum(h["faceCount"] // 3 for h in body.meshes if h["materialIdx"] in hair_mats)
    # 2 visible triangles, doubled for the double-sided material; the transparent quad is gone.
    assert hair_faces == 4


def test_materials_link_up(built):
    result, _, _ = built
    body = M.read_mod3(result.files["pl\\f_equip\\pl001_0000\\body\\mod\\f_body001.mod3"])
    mrl = R.read_mrl3(result.files["pl\\f_equip\\pl001_0000\\body\\mod\\f_body001.mrl3"])
    assert {jamcrc(n) for n in body.materials} == {m.name_hash for m in mrl.materials}
    paths = {p.lower() for p in result.files}
    for m in mrl.materials:
        roles = {}
        for b in m.bindings:
            if b.res_type & 0xF == R.RES_TEXTURE:
                assert 1 <= b.tex_idx <= len(mrl.textures)
                roles[R.slot_role(b.map_type)[0]] = mrl.textures[b.tex_idx - 1][2]
        for role in ("albedo", "normal", "rmt", "black"):
            assert (roles[role] + ".tex").lower() in paths, role
        # Unknown slots keep pointing at the original game texture.
        assert roles["unknown"].endswith("common_special")
        assert m.params == struct.pack("<16f", *([1.0] * 16))


def test_blank_models_are_valid(built):
    result, files, _ = built
    helm = M.read_mod3(result.files["pl\\f_equip\\pl001_0000\\helm\\mod\\f_helm001.mod3"])
    assert helm.header["meshCount"] == 1
    assert helm.materials == ["Ch_Pl_Helm_Mt"]
    pos, nrm, w, ids, faces = decode_vertices(helm, 0)
    assert len(pos) == 3 and np.allclose(pos, pos[0])


def test_armor_listing(tmp_path):
    make_game_folder(str(tmp_path))
    assert armor_sets(FolderFiles(str(tmp_path))) == ["pl001_0000"]


def test_split_large_mesh():
    from vroidhunter.build import _split
    n = 70000
    pos = np.random.rand(n, 3)
    faces = np.arange(n - n % 3).reshape(-1, 3)
    pieces = _split(pos, pos, np.hstack([pos, np.ones((n, 1))]), pos[:, :2], np.zeros((n, 4), int),
                    np.tile([1.0, 0, 0, 0], (n, 1)), faces)
    assert len(pieces) == 2
    assert all(len(p[0]) <= 0xFFFF for p in pieces)
    assert sum(len(p[6]) for p in pieces) == len(faces)


def test_bc1_roundtrip_colours():
    rgba = np.zeros((8, 8, 4), np.uint8)
    rgba[:, :, 0] = 255
    rgba[:, :, 3] = 255
    data = X.encode_bc1(rgba)
    assert len(data) == 4 * 8
    c0 = struct.unpack_from("<H", data, 0)[0]
    assert (c0 >> 11) == 31  # red endpoint


def test_partial_body_skeleton_moves_legs_to_another_piece(tmp_path):
    data, _ = make_vrm(0)
    (tmp_path / "a.vrm").write_bytes(data)
    make_game_folder(str(tmp_path / "g"), genders=("f",), partial_body=True)
    files = FolderFiles(str(tmp_path / "g"))
    result = build(str(tmp_path / "a.vrm"), files, Options(genders=("f",)), log=lambda m: None)
    used_anywhere = set()
    for part in ("helm", "body", "arm", "wst", "leg"):
        path = "pl\\f_equip\\pl001_0000\\%s\\mod\\f_%s001.mod3" % (part, part)
        mod = M.read_mod3(result.files[path])
        funcs = {b.function for b in mod.bones}
        for i in range(mod.header["meshCount"]):
            _, _, w, ids, _ = decode_vertices(mod, i)
            used = {mod.bones[j].function for j in ids[w > 0]}
            assert used <= funcs, part
            if mod.header["vertexCount"] > 3:
                used_anywhere |= used
                assert path.replace(".mod3", ".mrl3") in result.files
    assert {15, 19, 7, 11} <= used_anywhere
