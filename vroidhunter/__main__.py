"""Entry point.

No arguments (or just a .vrm path, e.g. by dropping the file on the .exe)
opens the window. Subcommands give a command line:

    VRoidHunter install  model.vrm [--game DIR] [--armor pl123_0000] [--second-armor pl067_0000|none] [--no-physics] [--height 1.0]
    VRoidHunter export   model.vrm OUTDIR [same options]
    VRoidHunter uninstall [--game DIR]
    VRoidHunter armors   [--game DIR]
"""

import argparse
import os
import sys


def _cli(argv):
    from . import game as G
    from .build import DEFAULT_ARMOR, DEFAULT_SECOND_ARMOR, FULL_HELMET, Options, build, scan_armor
    from . import oodle as O
    from .chunk import FolderFiles, GameFiles

    ap = argparse.ArgumentParser(prog="VRoidHunter")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, vrm=True):
        if vrm:
            p.add_argument("vrm")
        p.add_argument("--game", default=None, help="folder containing MonsterHunterWorld.exe")
        p.add_argument("--extracted", default=None,
                       help="use a folder of already-extracted game files instead of reading chunks")
        p.add_argument("--armor", default=DEFAULT_ARMOR)
        p.add_argument("--second-armor", default=DEFAULT_SECOND_ARMOR,
                       help="a second armor set that also becomes the avatar ('none' to skip)")
        p.add_argument("--no-physics", action="store_true", help="don't convert spring bones to physics")
        p.add_argument("--no-neck-cover", action="store_true",
                       help="don't thicken the avatar's neck to hide the hunter's own neck")
        p.add_argument("--hunter-proportions", action="store_true",
                       help="stretch the avatar onto the hunter's build instead of keeping its proportions")
        p.add_argument("--oodle", default=None, help="path to oo2core_8_win64.dll (or oo2core_9)")
        p.add_argument("--download-oodle", action="store_true",
                       help="if oo2core_8_win64.dll isn't found, download it from Warframe's public update server")
        p.add_argument("--female-only", action="store_true")
        p.add_argument("--height", type=float, default=1.0, help="height multiplier (1.0 = hunter height)")

    common(sub.add_parser("install"))
    ex = sub.add_parser("export")
    common(ex)
    ex.add_argument("out")
    common(sub.add_parser("uninstall"), vrm=False)
    common(sub.add_parser("armors"), vrm=False)
    args = ap.parse_args(argv)

    game_dir = args.game or G.find_game()

    def files():
        if args.extracted:
            return FolderFiles(args.extracted)
        if not G.is_game_dir(game_dir):
            sys.exit("Could not find Monster Hunter: World. Pass --game \"C:\\...\\Monster Hunter World\".")
        dll = args.oodle or O.find_oodle(game_dir, log=print)
        if not dll and args.download_oodle:
            dll = O.download_oodle()
        if not dll:
            sys.exit("oo2core_8_win64.dll not found. MHW only ships an older copy that can't read its files. "
                     "Pass --oodle PATH (copy it from another game) or --download-oodle.")
        return GameFiles(game_dir, oodle_path=dll)

    if args.cmd == "uninstall":
        if not G.is_game_dir(game_dir):
            sys.exit("Could not find Monster Hunter: World. Pass --game.")
        G.uninstall(game_dir)
        return
    if args.cmd == "armors":
        gf = files()
        for armor, cov in scan_armor(gf, "f", log=lambda m: None):
            print(armor + ("  full helmet" if cov is not None and cov >= FULL_HELMET else ""))
        return

    second = "" if args.second_armor.lower() in ("", "none") else args.second_armor
    opts = Options(armor=args.armor, second_armor=second, genders=("f",) if args.female_only else ("f", "m"),
                   physics=not args.no_physics, keep_proportions=not args.hunter_proportions,
                   cover_neck=not args.no_neck_cover,
                   height_scale=args.height)
    gf = files()
    try:
        result = build(args.vrm, gf, opts)
    finally:
        gf.close()
    if args.cmd == "export":
        G.export_folder(args.out, result.files)
        print("Exported %d files to %s" % (len(result.files), os.path.join(args.out, "nativePC")))
    else:
        if not G.is_game_dir(game_dir):
            sys.exit("Could not find Monster Hunter: World. Pass --game.")
        G.install(game_dir, result.files, {"vrm": os.path.basename(args.vrm), "armor": args.armor,
                                                   "second_armor": second})
        if not G.loader_installed(game_dir):
            print("WARNING: Stracker's Loader is not installed; the game will not load nativePC mods without it.")


def _ensure_streams():
    """The windowed .exe has no console; send output to VRoidHunter.log instead."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    base = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    try:
        log = open(os.path.join(base, "VRoidHunter.log"), "w", encoding="utf-8", buffering=1)
    except OSError:
        log = open(os.devnull, "w")
    if sys.stdout is None:
        sys.stdout = log
    if sys.stderr is None:
        sys.stderr = log


def main():
    _ensure_streams()
    argv = sys.argv[1:]
    if argv and argv[0] in ("install", "export", "uninstall", "armors", "-h", "--help"):
        _cli(argv)
        return
    from .gui import main as gui_main
    gui_main(argv[0] if argv and os.path.isfile(argv[0]) else "")


if __name__ == "__main__":
    main()
