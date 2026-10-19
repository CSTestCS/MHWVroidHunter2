"""Locate (or fetch) the Oodle decompression library needed to read MHW's chunk files.

Monster Hunter: World stores its files compressed with Oodle, but the game
itself only ships an old Oodle (oo2core_5) that cannot unpack the current
Iceborne archives. Many other games ship the newer oo2core_8/oo2core_9
library, so we look for one on the PC first. If none is found, the user can
choose to download the copy Warframe's publicly hosted update server provides
(the same source the community tool MHWNoChunk uses), checked against known
MD5 hashes before it is used.
"""

import hashlib
import io
import lzma
import os
import urllib.request

from .game import _library_folders, _steam_roots

NAMES = ("oo2core_9_win64.dll", "oo2core_8_win64.dll")
DOWNLOAD_URL = ("https://origin.warframe.com/origin/00000000/Tools/Oodle/x64/final/"
                "oo2core_8_win64.dll.3169B48A9A2086E53C4493C03579902C.lzma")
# Known good oo2core_8_win64.dll builds (from MHWNoChunk, MIT licence).
KNOWN_MD5 = {
    "9b7f9e3e4931b80257da5e1b626db43a", "c1a0dd317543035221327d44f07c3d06",
    "224df07f2fc2bd829f3b221bb5ef1a31", "b486c6f46a3d802966d04911a619b2ed",
    "548800ca453904c9a892521a64d71f73", "3169b48a9a2086e53c4493c03579902c",
}


def cache_dir():
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "VRoidHunter")


def _exe_dir():
    import sys
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.getcwd()


def _check(folder):
    for name in NAMES:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    return None


def find_oodle(game_dir=None, search_other_games=True, log=lambda m: None):
    """Path of a usable oo2core_8/9 DLL, or None."""
    for folder in filter(None, [cache_dir(), _exe_dir(), game_dir]):
        found = _check(folder)
        if found:
            return found
    if not search_other_games:
        return None
    log("Looking for oo2core_8/9_win64.dll in your other installed games...")
    roots = set()
    for root in _steam_roots():
        if root and os.path.isdir(root):
            for lib in _library_folders(root):
                roots.add(os.path.join(lib, "steamapps", "common"))
    for drive in "CDEFGH":
        roots.add(drive + r":\Program Files\Epic Games")
        roots.add(drive + r":\XboxGames")
    for root in sorted(roots):
        if not os.path.isdir(root):
            continue
        for game in os.listdir(root):
            top = os.path.join(root, game)
            # Search a few levels deep; these DLLs sit near the game exe.
            for dirpath, dirnames, filenames in os.walk(top):
                depth = dirpath[len(top):].count(os.sep)
                if depth >= 4:
                    dirnames[:] = []
                lower = {f.lower(): f for f in filenames}
                for name in NAMES:
                    if name in lower:
                        path = os.path.join(dirpath, lower[name])
                        log("Found %s" % path)
                        return path
    return None


def remember(path):
    """Copy a DLL the user picked into our cache so it is found next time."""
    os.makedirs(cache_dir(), exist_ok=True)
    target = os.path.join(cache_dir(), os.path.basename(path).lower())
    if os.path.abspath(path) != os.path.abspath(target):
        with open(path, "rb") as src, open(target, "wb") as dst:
            dst.write(src.read())
    return target


def download_oodle(log=print):
    """Download oo2core_8_win64.dll from Warframe's public update server and verify it."""
    log("Downloading oo2core_8_win64.dll from Warframe's public update server...")
    with urllib.request.urlopen(DOWNLOAD_URL, timeout=60) as response:
        packed = response.read()
    data = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE).decompress(packed)
    digest = hashlib.md5(data).hexdigest()
    if digest not in KNOWN_MD5:
        raise IOError("downloaded file did not match the expected checksum (%s)" % digest)
    if data[:2] != b"MZ":
        raise IOError("downloaded file is not a Windows DLL")
    os.makedirs(cache_dir(), exist_ok=True)
    path = os.path.join(cache_dir(), "oo2core_8_win64.dll")
    with open(path, "wb") as f:
        f.write(data)
    log("Saved to %s" % path)
    return path


def unpack_alone(packed):
    """Exposed for tests: decode a classic .lzma (LZMA-alone) payload."""
    return lzma.LZMADecompressor(format=lzma.FORMAT_ALONE).decompress(io.BytesIO(packed).read())
