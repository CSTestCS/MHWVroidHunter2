"""Entry point.

No arguments (or just a .vrm path, e.g. by dropping the file on the .exe)
opens the window. Subcommands give a command line:

    VRoidHunter install  model.vrm [--game DIR] [--armor pl001_0000] [--keep-face] [--height 1.0]
    VRoidHunter export   model.vrm OUTDIR [same options]
    VRoidHunter uninstall [--game DIR]
    VRoidHunter armors   [--game DIR]
"""

import argparse
import os
import sys


def _cli(argv):
    from . import game as G
    from .build import DEFAULT_ARMOR, Options, armor_sets, build
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
        p.add_argument("--keep-face", action="store_true", help="do not hide the hunter's own face and hair")
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
        return GameFiles(game_dir)

    if args.cmd == "uninstall":
        if not G.is_game_dir(game_dir):
            sys.exit("Could not find Monster Hunter: World. Pass --game.")
        G.uninstall(game_dir)
        return
    if args.cmd == "armors":
        gf = files()
        print("\n".join(armor_sets(gf)))
        return

    opts = Options(armor=args.armor, genders=("f",) if args.female_only else ("f", "m"),
                   hide_face_and_hair=not args.keep_face, height_scale=args.height)
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
        G.install(game_dir, result.files, {"vrm": os.path.basename(args.vrm), "armor": args.armor})
        if not G.loader_installed(game_dir):
            print("WARNING: Stracker's Loader is not installed; the game will not load nativePC mods without it.")


def main():
    argv = sys.argv[1:]
    if argv and argv[0] in ("install", "export", "uninstall", "armors", "-h", "--help"):
        _cli(argv)
        return
    from .gui import main as gui_main
    gui_main(argv[0] if argv and os.path.isfile(argv[0]) else "")


if __name__ == "__main__":
    main()
