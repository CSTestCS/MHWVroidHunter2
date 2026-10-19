import hashlib
import io
import lzma

import pytest

from vroidhunter import oodle as O


def test_finds_dll_in_cache_or_game_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    game = tmp_path / "game"
    game.mkdir()
    assert O.find_oodle(str(game), search_other_games=False) is None
    (game / "oo2core_9_win64.dll").write_bytes(b"MZ")
    assert O.find_oodle(str(game), search_other_games=False).endswith("oo2core_9_win64.dll")
    picked = tmp_path / "elsewhere" / "oo2core_8_win64.dll"
    picked.parent.mkdir()
    picked.write_bytes(b"MZ picked")
    cached = O.remember(str(picked))
    assert O.find_oodle(None, search_other_games=False) == cached


def _fake_server(monkeypatch, payload):
    packed = lzma.compress(payload, format=lzma.FORMAT_ALONE)

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(O.urllib.request, "urlopen", lambda url, timeout=0: Resp(packed))


def test_download_verifies_checksum(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    payload = b"MZ" + b"\x00" * 100
    _fake_server(monkeypatch, payload)
    with pytest.raises(IOError):
        O.download_oodle(log=lambda m: None)
    monkeypatch.setattr(O, "KNOWN_MD5", {hashlib.md5(payload).hexdigest()})
    path = O.download_oodle(log=lambda m: None)
    with open(path, "rb") as f:
        assert f.read() == payload
