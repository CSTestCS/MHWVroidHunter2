import struct

import numpy as np
import pytest

from vroidhunter import ctc as C
from vroidhunter import mod3 as M
from vroidhunter import mrl3 as R
from vroidhunter import tex as X
from vroidhunter.binio import jamcrc
from vroidhunter.build import Options, armor_sets, build
from vroidhunter.chunk import FolderFiles
from vroidhunter.vrm import load_vrm_bytes

from synth import hunter_bones, make_game_folder, make_hunter_mod3, make_vrm


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
    result = build(str(vrm_path), files, Options(armor=ARMOR), log=lambda m: None)
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


ARMOR = "pl001_0000"
BASE = "pl\\f_equip\\pl001_0000\\%s\\mod\\f_%s001.%s"


def piece(result, part, ext="mod3"):
    return result.files.get(BASE % (part, part, ext))


def gather(mod):
    """All vertices of a model with their main bone function."""
    out = []
    for i in range(mod.header["meshCount"]):
        pos, nrm, w, ids, faces = decode_vertices(mod, i)
        funcs = np.vectorize(lambda j: mod.bones[j].function)(ids)
        out.append((i, pos, nrm, w, funcs, faces))
    return out


def test_expected_files(built):
    result, files, _ = built
    paths = set(result.files)
    for g in ("f", "m"):
        for part in ("helm", "body", "arm", "wst", "leg"):
            assert "pl\\%s_equip\\pl001_0000\\%s\\mod\\%s_%s001.mod3" % (g, part, g, part) in paths
        for part in ("helm", "body"):
            assert "pl\\%s_equip\\pl001_0000\\%s\\mod\\%s_%s001.mrl3" % (g, part, g, part) in paths
    # Nothing outside the chosen armor set is touched: no faces, no hair, no other gear.
    for p in paths:
        assert p.startswith("pl\\f_equip\\pl001_0000\\") or p.startswith("pl\\m_equip\\pl001_0000\\"), p
    texes = [p for p in paths if p.endswith(".tex")]
    assert texes, "no textures written"
    for p in texes:
        mips, w, h, fmt, offsets = X.read_tex_header(result.files[p])
        assert w >= 4 and h >= 4 and mips >= 1
        assert offsets[0] == 0xB8 + 8 * mips
        assert list(offsets) == sorted(offsets)
        assert offsets[-1] < len(result.files[p])


def test_avatar_fits_hunter(built):
    result, files, flip = built
    body = M.read_mod3(piece(result, "body"))
    helm = M.read_mod3(piece(result, "helm"))
    ref = M.read_mod3(files.read(BASE % ("body", "body", "mod3")))
    # Same bones and hierarchy as the game's model (joints may move to fit the avatar).
    assert [(b.function, b.parent) for b in body.bones] == [(b.function, b.parent) for b in ref.bones]
    assert body.remap == ref.remap
    bones = hunter_bones()
    all_pos, all_funcs, body_z = [], [], []
    for mod in (body, helm):
        for i, pos, nrm, w, funcs, faces in gather(mod):
            assert faces.min() >= 0 and faces.max() < len(pos)
            assert np.all(w >= -1e-3) and np.allclose(w.sum(axis=1), 1, atol=1e-6)
            all_pos.append(pos)
            all_funcs.append(funcs[:, 0])
            if "Hair" not in mod.materials[mod.meshes[i]["materialIdx"]]:
                body_z.append(pos[:, 2])
            fn = np.cross(pos[faces[:, 1]] - pos[faces[:, 0]], pos[faces[:, 2]] - pos[faces[:, 0]])
            agree = np.sum(fn * nrm[faces].sum(axis=1), axis=1)
            assert (np.mean(agree > 0) if not flip else np.mean(agree < 0)) > 0.9
    pos = np.vstack(all_pos)
    funcs = np.concatenate(all_funcs)
    z = np.concatenate(body_z)
    assert z.max() == pytest.approx(156, abs=6)
    assert z.min() == pytest.approx(9 - 6, abs=8)
    for f in (3, 7, 8, 15, 19):
        pts = pos[funcs == f]
        assert len(pts), "no vertices weighted to bone %d" % f
        assert np.linalg.norm(pts - np.array(bones[f][1]), axis=1).min() < 10, f
    assert np.mean(pos[funcs == 8][:, 0]) < 0
    assert np.mean(pos[funcs == 12][:, 0]) > 0
    assert np.mean(pos[funcs == 16][:, 1]) < -2


def test_head_and_hair_go_in_the_helmet(built):
    result, _, _ = built
    helm = M.read_mod3(piece(result, "helm"))
    body = M.read_mod3(piece(result, "body"))
    assert any("Hair" in n for n in helm.materials)
    assert not any("Hair" in n for n in body.materials)
    for _, pos, _, w, funcs, _ in gather(helm):
        used = set(funcs[w > 0].ravel())
        assert used <= {4} | set(range(150, 247)), used


def test_spring_bones_become_physics(built):
    result, _, _ = built
    helm = M.read_mod3(piece(result, "helm"))
    by_func = {b.function: (i, b) for i, b in enumerate(helm.bones)}
    ctc = piece(result, "helm", "ctc")
    assert ctc is not None
    chains = C.read_chains(ctc)
    assert len(chains) == 1
    fixed, funcs = chains[0]
    assert len(funcs) == 3 and fixed[0] == 1 and all(f >= 150 for f in funcs)
    # Chain bones exist in the skeleton, parented root -> head bone, then along the chain.
    root_parent = helm.bones[by_func[funcs[0]][1].parent].function
    assert root_parent == 4
    assert helm.bones[by_func[funcs[1]][1].parent].function == funcs[0]
    assert helm.bones[by_func[funcs[2]][1].parent].function == funcs[1]
    # The hair card is weighted to the physics bones; the tip bone carries no weight.
    hair_funcs = set()
    for i, pos, _, w, f, _ in gather(helm):
        if "Hair" in helm.materials[helm.meshes[i]["materialIdx"]]:
            hair_funcs |= set(f[w > 0].ravel())
    assert hair_funcs == {funcs[0], funcs[1]}
    # Skeleton stays consistent: world matrix of each new bone = inverse of its AMatrix.
    worlds = helm.bone_world_matrices()
    for f in funcs:
        i = by_func[f][0]
        parent_world = worlds[helm.bones[i].parent]
        assert np.allclose(parent_world @ helm.lmatrices[i], worlds[i], atol=1e-3)
    # The fitted bone sits above the hunter's head bone.
    assert worlds[by_func[funcs[0]][0]][2, 3] > hunter_bones()[4][1][2]


def test_physics_can_be_turned_off(tmp_path):
    data, _ = make_vrm(0)
    (tmp_path / "a.vrm").write_bytes(data)
    make_game_folder(str(tmp_path / "g"), genders=("f",))
    result = build(str(tmp_path / "a.vrm"), FolderFiles(str(tmp_path / "g")),
                   Options(armor=ARMOR, genders=("f",), physics=False), log=lambda m: None)
    assert not any(p.endswith(".ctc") for p in result.files)


def test_transparent_triangles_culled(built):
    result, _, _ = built
    helm = M.read_mod3(piece(result, "helm"))
    hair_mats = [i for i, n in enumerate(helm.materials) if "Hair" in n]
    hair_faces = sum(h["faceCount"] // 3 for h in helm.meshes if h["materialIdx"] in hair_mats)
    # 2 visible triangles, doubled for the double-sided material; the transparent quad is gone.
    assert hair_faces == 4


def test_materials_link_up(built):
    result, _, _ = built
    paths = {p.lower() for p in result.files}
    for part in ("body", "helm"):
        mod = M.read_mod3(piece(result, part))
        mrl = R.read_mrl3(piece(result, part, "mrl3"))
        assert {jamcrc(n) for n in mod.materials} == {m.name_hash for m in mrl.materials}
        for m in mrl.materials:
            roles = {}
            for b in m.bindings:
                if b.res_type & 0xF == R.RES_TEXTURE:
                    assert 1 <= b.tex_idx <= len(mrl.textures)
                    roles[R.slot_role(b.map_type)[0]] = mrl.textures[b.tex_idx - 1][2]
            for role in ("albedo", "normal", "rmt", "black"):
                assert (roles[role] + ".tex").lower() in paths, role
            assert roles["unknown"].endswith("common_special")
            # Cloned from the plain player material (not the shiny metal one), and set up as
            # a non-metal, non-dyeable surface that still gets the game's wetness.
            assert m.fields[2] == 0xBEEF0002
            assert "CBMhMaterialGlobal" in R.cbuffer_names(m)
            assert R.get_field(m, "fMetalic__uiUNorm") == 0.0
            assert R.get_field(m, "fFilmBlend__uiUNorm") == 0.0
            assert R.get_field(m, "bUseCMM") == 0
            assert R.get_field(m, "fMaterialWetBlend") == pytest.approx(0.25)   # left to the game
        kinds = {n: R.get_field(m, "fSubSurfaceBlend__uiUNorm") for n, m in
                 zip(mod.materials, sorted(mrl.materials, key=lambda m: mod.materials.index(
                     next(x for x in mod.materials if jamcrc(x) == m.name_hash))))}
        for n, sss in kinds.items():
            assert sss == (0.5 if "SKIN" in n.upper() else 0.0), n


def test_unused_pieces_are_invisible(built):
    result, files, _ = built
    arm = M.read_mod3(piece(result, "arm"))
    assert arm.header["meshCount"] == 1
    assert arm.materials == ["Ch_Pl_Arm_Mt"]
    pos, nrm, w, ids, faces = decode_vertices(arm, 0)
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
    result = build(str(tmp_path / "a.vrm"), files, Options(armor=ARMOR, genders=("f",)), log=lambda m: None)
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


def test_helm_coverage_tells_full_from_open_helmets():
    from vroidhunter.build import helm_coverage
    bones = hunter_bones()
    # The fake hunter faces -Y; its face is a small box in front of the head bone.
    face = M.read_mod3(make_hunter_mod3(bones, box=((-6, 6), (-9, -3), (152, 166))))
    face_pts = np.vstack([face.mesh_geometry(0)[0]])
    full = M.read_mod3(make_hunter_mod3(bones, box=((-11, 11), (-13, 11), (146, 175))))
    cap = M.read_mod3(make_hunter_mod3(bones, box=((-11, 11), (-8, 11), (167, 178))))
    assert helm_coverage(full, face_pts) > 0.8
    assert helm_coverage(cap, face_pts) < 0.5


def test_physics_budget_thins_long_chains():
    from vroidhunter.build import _plan_physics
    from vroidhunter.vrm import SpringChain

    class Prim:
        joints = np.arange(200).reshape(-1, 1)
        weights = np.ones((200, 1))

    class Model:
        springs = [SpringChain(list(range(0, 120))), SpringChain(list(range(120, 125)))]
        primitives = [Prim()]

    planned, node_funcs = _plan_physics(Model(), set(range(150, 160)), 255, lambda m: None)
    used = sum(len(c.nodes) + 1 for c in planned)
    assert used <= 97 - 10
    # Every spring joint still follows some physics bone, and ids are unique and free.
    assert set(range(125)) <= set(node_funcs)
    ids = [f for c in planned for f in c.funcs + [c.end_func]]
    assert len(ids) == len(set(ids)) and not set(ids) & set(range(150, 160))
    assert all(c.nodes[0] in (0, 120) for c in planned)


def _limb_check(tmp_path, keep):
    # An avatar with long legs and long forearms, unlike the hunter.
    data, joints = make_vrm(0, leg_scale=1.35, arm_scale=1.3)
    (tmp_path / "a.vrm").write_bytes(data)
    make_game_folder(str(tmp_path / "g"), genders=("f",))
    files = FolderFiles(str(tmp_path / "g"))
    result = build(str(tmp_path / "a.vrm"), files, Options(armor=ARMOR, genders=("f",), keep_proportions=keep),
                   log=lambda m: None)
    body = M.read_mod3(piece(result, "body"))
    worlds = body.bone_world_matrices()
    at = {b.function: worlds[i][:3, 3] for i, b in enumerate(body.bones)}
    return body, at, joints


def _ratio(a, b, c, d):
    return np.linalg.norm(a - b) / np.linalg.norm(c - d)


def test_keeps_avatar_proportions(tmp_path):
    body, at, j = _limb_check(tmp_path, keep=True)
    # Leg / torso ratio of the skeleton now matches the avatar, not the hunter.
    avatar = _ratio(j["leftUpperLeg"], j["leftFoot"], (j["leftUpperLeg"] + j["rightUpperLeg"]) / 2, j["neck"])
    skel = _ratio(at[14], at[16], (at[14] + at[18]) / 2, at[3])
    hunter = hunter_bones()
    hunter_ratio = _ratio(np.array(hunter[14][1]), np.array(hunter[16][1]),
                          (np.array(hunter[14][1]) + np.array(hunter[18][1])) / 2, np.array(hunter[3][1]))
    assert skel == pytest.approx(avatar, rel=0.08)
    assert abs(skel - hunter_ratio) > 0.2
    # Forearms are the one exception: they take the hunter's length so wrists line up.
    hunter_fore = np.linalg.norm(np.array(hunter[8][1]) - np.array(hunter[7][1]))
    assert np.linalg.norm(at[8] - at[7]) == pytest.approx(hunter_fore, rel=0.01)
    assert np.linalg.norm(at[12] - at[11]) == pytest.approx(hunter_fore, rel=0.01)
    # The forearm twist bone sits half-way along the moved forearm, like on the hunter.
    assert np.allclose(at[81], (at[7] + at[8]) / 2, atol=0.5)
    # Still the hunter's height, standing on the ground, arms in the hunter's A-pose direction.
    assert at[4][2] == pytest.approx(hunter[4][1][2], abs=4)
    assert (at[16][2] + at[20][2]) / 2 == pytest.approx((hunter[16][1][2] + hunter[20][1][2]) / 2, abs=2)
    arm = (at[8] - at[6]) / np.linalg.norm(at[8] - at[6])
    hunter_arm = np.array(hunter[8][1]) - np.array(hunter[6][1])
    assert np.dot(arm, hunter_arm / np.linalg.norm(hunter_arm)) > 0.99
    # Skin follows the moved bones: calf vertices sit around the moved calf joint.
    for _, pos, _, w, f, _ in gather(body):
        calf = pos[f[:, 0] == 15]
        if len(calf):
            assert np.linalg.norm(calf - at[15], axis=1).min() < 6
    # The upper arm keeps the avatar's own length (relative to the torso).
    assert _ratio(at[6], at[7], (at[14] + at[18]) / 2, at[3]) == pytest.approx(
        _ratio(j["leftUpperArm"], j["leftLowerArm"], (j["leftUpperLeg"] + j["rightUpperLeg"]) / 2, j["neck"]),
        rel=0.08)


def test_hunter_proportions_option_leaves_skeleton(tmp_path):
    body, at, j = _limb_check(tmp_path, keep=False)
    hunter = hunter_bones()
    for f in (6, 7, 8, 14, 15, 16):
        assert np.allclose(at[f], hunter[f][1], atol=1e-3)


def test_physics_uses_the_avatars_own_colliders(built):
    from vroidhunter import ccl as CL
    result, files, _ = built
    helm = M.read_mod3(piece(result, "helm"))
    ccl = piece(result, "helm", "ccl")
    assert ccl is not None, "our own collision file replaces the armor's"
    caps = CL.read_ccl(ccl)
    assert len(caps) == 1
    fa, oa, ra, fb, ob, rb = caps[0]
    assert fa == fb == 4                    # the VRM's head collider, on the head bone
    scale = 147 / 1.40                      # hunter / avatar ankle-to-head height
    assert ra == pytest.approx(0.09 * scale, rel=0.05)
    worlds = helm.bone_world_matrices()
    head = worlds[helm.function_to_index()[4]]
    centre = (head @ np.append(oa, 1))[:3]
    assert centre[2] == pytest.approx(head[2, 3] + 0.08 * scale, abs=2)
    # Chains collide with it, and each node's frame points along the chain.
    rec = piece(result, "helm", "ctc")
    assert rec[C.HEADER_SIZE + C.CHAIN_COLLISION] == 4
    nodes_at = C.HEADER_SIZE + C.CHAIN_SIZE
    node_size = (len(rec) - nodes_at) // 3
    by_func = helm.function_to_index()
    fixed, funcs = C.read_chains(rec)[0]
    for i in range(2):
        m = np.array(struct.unpack_from("<16f", rec, nodes_at + node_size * i)).reshape(4, 4).T
        a, b = worlds[by_func[funcs[i]]][:3, 3], worlds[by_func[funcs[i + 1]]][:3, 3]
        assert np.dot(m[:3, 0], (b - a) / np.linalg.norm(b - a)) == pytest.approx(1, abs=1e-4)


def test_auto_capsules_without_vrm_colliders(tmp_path):
    from vroidhunter import ccl as CL
    data, _ = make_vrm(0)
    (tmp_path / "a.vrm").write_bytes(data)
    make_game_folder(str(tmp_path / "g"), genders=("f",))
    import vroidhunter.build as B
    real = B.load_vrm

    def no_colliders(path):
        m = real(path)
        m.colliders = []
        return m

    B.load_vrm = no_colliders
    try:
        result = build(str(tmp_path / "a.vrm"), FolderFiles(str(tmp_path / "g")), Options(armor=ARMOR, genders=("f",)),
                       log=lambda m: None)
    finally:
        B.load_vrm = real
    caps = CL.read_ccl(piece(result, "helm", "ccl"))
    bones = {c[0] for c in caps}
    assert {2, 6, 10, 14} <= bones
    assert all(c[2] > 0 for c in caps)


def test_neck_covers_the_hunters_neck(tmp_path):
    data, _ = make_vrm(0)
    (tmp_path / "a.vrm").write_bytes(data)
    make_game_folder(str(tmp_path / "g"), genders=("f",))
    files = FolderFiles(str(tmp_path / "g"))
    on = build(str(tmp_path / "a.vrm"), files, Options(armor=ARMOR, genders=("f",)), log=lambda m: None)
    off = build(str(tmp_path / "a.vrm"), files, Options(armor=ARMOR, genders=("f",), cover_neck=False), log=lambda m: None)

    def neck_radii(result):
        body = M.read_mod3(piece(result, "body"))
        w = body.bone_world_matrices()
        idx = body.function_to_index()
        a, b = w[idx[3]][:3, 3], w[idx[4]][:3, 3]
        axis = (b - a) / np.linalg.norm(b - a)
        out = []
        for _, pos, _, wt, f, _ in gather(body):
            sel = f[:, 0] == 3
            d = pos[sel] - a
            h = d @ axis
            r = np.linalg.norm(d - np.outer(h, axis), axis=1)
            out.append(r[(h > 0) & (h < 4)])
        return np.concatenate(out)

    before, after = neck_radii(off), neck_radii(on)
    assert before.max() < 4            # the avatar's neck is thinner than the hunter's (5 units)
    # Now it covers it, but never grows more than 2x (the test avatar's neck is very thin).
    assert after.min() >= min(5 * 0.95, before.min() * 2.0 - 1e-6)
    assert after.max() <= before.max() * 2.0 + 1e-6
    # Nothing far from the neck moves.
    b_off = M.read_mod3(piece(off, "body"))
    b_on = M.read_mod3(piece(on, "body"))
    for (_, p0, _, _, f0, _), (_, p1, _, _, f1, _) in zip(gather(b_off), gather(b_on)):
        far = f0[:, 0] != 3
        assert np.allclose(p0[far], p1[far])


def test_arms_copy_the_game_armors_twist_weighting(built):
    result, _, _ = built
    body = M.read_mod3(piece(result, "body"))
    w = body.bone_world_matrices()
    idx = body.function_to_index()
    checked = 0
    for fore, hand, twist in ((7, 8, 81), (11, 12, 83)):
        a, b = w[idx[fore]][:3, 3], w[idx[hand]][:3, 3]
        ab = b - a
        for _, pos, _, wt, f, _ in gather(body):
            t = (pos - a) @ ab / (ab @ ab)
            d = np.linalg.norm(pos - (a + np.outer(t, ab)), axis=1)
            on = (t > 0.1) & (t < 0.9) & (d < 5) & np.isin(f[:, 0], [fore, twist])
            for i in np.nonzero(on)[0]:
                share = wt[i][f[i] == twist].sum()
                # The reference armor puts share t on the twist bone (binned, so allow slack).
                assert share == pytest.approx(t[i], abs=0.12), (t[i], share)
                checked += 1
    assert checked > 10


def test_face_details_are_not_skin():
    from vroidhunter.build import material_kind
    for name in ("N00_000_00_FaceBrow_00_FACE", "N00_000_00_FaceEyelash_00_FACE", "N00_000_00_FaceMouth_00_FACE"):
        assert material_kind(name) != "skin", name
    assert material_kind("N00_000_00_Face_00_SKIN") == "skin"
    assert material_kind("N00_000_00_Body_00_SKIN") == "skin"


def test_second_armor_set_also_becomes_the_avatar(tmp_path):
    data, _ = make_vrm(0)
    (tmp_path / "a.vrm").write_bytes(data)
    make_game_folder(str(tmp_path / "g"), genders=("f",))
    make_game_folder(str(tmp_path / "g"), armor="pl067_0000", genders=("f",))
    files = FolderFiles(str(tmp_path / "g"))
    one = build(str(tmp_path / "a.vrm"), files, Options(armor=ARMOR, second_armor="", genders=("f",)),
                log=lambda m: None)
    both = build(str(tmp_path / "a.vrm"), files, Options(armor=ARMOR, second_armor="pl067_0000", genders=("f",)),
                 log=lambda m: None)
    assert not any("pl067_0000" in p for p in one.files)
    for armor, num in ((ARMOR, "001"), ("pl067_0000", "067")):
        body = M.read_mod3(both.files["pl\\f_equip\\%s\\body\\mod\\f_body%s.mod3" % (armor, num)])
        assert body.header["vertexCount"] > 0
    # A missing second set is skipped, not an error.
    build(str(tmp_path / "a.vrm"), files, Options(armor=ARMOR, second_armor="pl999_0000", genders=("f",)),
          log=lambda m: None)
