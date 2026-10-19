import os

from vroidhunter import game as G


def test_install_and_uninstall_restore_other_mods(tmp_path):
    gd = tmp_path / "Monster Hunter World"
    (gd / "nativePC" / "pl").mkdir(parents=True)
    (gd / G.EXE_NAME).write_bytes(b"")
    other = gd / "nativePC" / "pl" / "f_body001.mod3"
    other.write_bytes(b"someone else's mod")
    files = {"pl\\f_body001.mod3": b"ours", "pl\\new\\x.tex": b"tex"}
    G.install(str(gd), files, {"vrm": "a.vrm"}, log=lambda m: None)
    assert other.read_bytes() == b"ours"
    assert (gd / "nativePC" / "pl" / "new" / "x.tex").read_bytes() == b"tex"
    # Reinstalling replaces our previous install cleanly.
    G.install(str(gd), {"pl\\f_body001.mod3": b"ours v2"}, {"vrm": "b.vrm"}, log=lambda m: None)
    assert other.read_bytes() == b"ours v2"
    assert not (gd / "nativePC" / "pl" / "new").exists()
    G.uninstall(str(gd), log=lambda m: None)
    assert other.read_bytes() == b"someone else's mod"
    assert not (gd / "nativePC" / G.MANIFEST).exists()
    assert not (gd / "nativePC" / G.BACKUP_DIR).exists()
