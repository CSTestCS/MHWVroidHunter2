"""Find the Steam install of Monster Hunter: World and install files into nativePC."""

import json
import os
import re
import shutil
import time

APP_ID = "582010"
EXE_NAME = "MonsterHunterWorld.exe"
MANIFEST = "vroidhunter_manifest.json"
BACKUP_DIR = "vroidhunter_backup"


def _steam_roots():
    roots = []
    if os.name == "nt":
        try:
            import winreg
            for hive, key, value in (
                    (winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath"),
                    (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath"),
                    (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Valve\Steam", "InstallPath")):
                try:
                    with winreg.OpenKey(hive, key) as k:
                        roots.append(winreg.QueryValueEx(k, value)[0])
                except OSError:
                    pass
        except ImportError:
            pass
        for drive in "CDEFGH":
            roots += [drive + r":\Program Files (x86)\Steam", drive + r":\Program Files\Steam",
                      drive + r":\Steam", drive + r":\SteamLibrary"]
    else:
        home = os.path.expanduser("~")
        roots += [os.path.join(home, ".steam", "steam"), os.path.join(home, ".local", "share", "Steam")]
    return roots


def _library_folders(steam_root):
    libs = [steam_root]
    vdf = os.path.join(steam_root, "steamapps", "libraryfolders.vdf")
    if os.path.exists(vdf):
        try:
            with open(vdf, encoding="utf-8", errors="replace") as f:
                text = f.read()
            for m in re.finditer(r'"path"\s+"([^"]+)"', text):
                libs.append(m.group(1).replace("\\\\", "\\"))
        except OSError:
            pass
    return libs


def find_game():
    """Return the game folder (containing MonsterHunterWorld.exe) or None."""
    seen = set()
    for root in _steam_roots():
        if not root or not os.path.isdir(root):
            continue
        for lib in _library_folders(root):
            if lib in seen:
                continue
            seen.add(lib)
            install = "Monster Hunter World"
            acf = os.path.join(lib, "steamapps", "appmanifest_%s.acf" % APP_ID)
            if os.path.exists(acf):
                try:
                    with open(acf, encoding="utf-8", errors="replace") as f:
                        m = re.search(r'"installdir"\s+"([^"]+)"', f.read())
                    if m:
                        install = m.group(1)
                except OSError:
                    pass
            path = os.path.join(lib, "steamapps", "common", install)
            if os.path.exists(os.path.join(path, EXE_NAME)):
                return path
    return None


def is_game_dir(path):
    return bool(path) and os.path.exists(os.path.join(path, EXE_NAME))


def loader_installed(game_dir):
    """Stracker's Loader puts loader.dll (and dinput8.dll) next to the game exe."""
    return os.path.exists(os.path.join(game_dir, "loader.dll")) or \
        os.path.exists(os.path.join(game_dir, "dinput8.dll"))


def _native(game_dir, rel):
    return os.path.join(game_dir, "nativePC", *rel.replace("/", "\\").split("\\"))


def read_manifest(game_dir):
    path = os.path.join(game_dir, "nativePC", MANIFEST)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def uninstall(game_dir, log=print):
    manifest = read_manifest(game_dir)
    if not manifest:
        log("Nothing to uninstall.")
        return 0
    removed = 0
    for rel in manifest.get("files", []):
        target = _native(game_dir, rel)
        if os.path.exists(target):
            os.remove(target)
            removed += 1
        backup = os.path.join(game_dir, "nativePC", BACKUP_DIR, *rel.split("\\"))
        if os.path.exists(backup):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.move(backup, target)
    backup_root = os.path.join(game_dir, "nativePC", BACKUP_DIR)
    if os.path.isdir(backup_root):
        shutil.rmtree(backup_root, ignore_errors=True)
    os.remove(os.path.join(game_dir, "nativePC", MANIFEST))
    # Tidy up directories we created and left empty.
    for rel in manifest.get("files", []):
        d = os.path.dirname(_native(game_dir, rel))
        while d.startswith(os.path.join(game_dir, "nativePC")) and d != os.path.join(game_dir, "nativePC"):
            try:
                os.rmdir(d)
            except OSError:
                break
            d = os.path.dirname(d)
    log("Removed %d files." % removed)
    return removed


def install(game_dir, files, info, log=print):
    """Write {relative path: bytes} into nativePC, backing up anything we replace."""
    if read_manifest(game_dir):
        log("Removing the previous VRoid Hunter install first.")
        uninstall(game_dir, log)
    written = []
    for rel, data in sorted(files.items()):
        target = _native(game_dir, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.exists(target):
            backup = os.path.join(game_dir, "nativePC", BACKUP_DIR, *rel.split("\\"))
            os.makedirs(os.path.dirname(backup), exist_ok=True)
            shutil.move(target, backup)
        with open(target, "wb") as f:
            f.write(data)
        written.append(rel)
    manifest = dict(info)
    manifest["files"] = written
    manifest["installed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(game_dir, "nativePC", MANIFEST), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log("Wrote %d files to %s" % (len(written), os.path.join(game_dir, "nativePC")))
    return written


def export_folder(out_dir, files):
    """Write the mod as a plain nativePC folder (for mod managers or manual copying)."""
    for rel, data in files.items():
        target = os.path.join(out_dir, "nativePC", *rel.split("\\"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as f:
            f.write(data)
