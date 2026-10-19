# VRoid Hunter

Use any `.vrm` avatar (VRoid Studio, VRM 0.x or 1.0) as your hunter's look in
**Monster Hunter: World** (Steam).

VRoid Hunter is a single `VRoidHunter.exe`. Pick your `.vrm`, pick an armor
set, press **Install**. When you wear that armor set, your hunter shows up as
your avatar, sized to your hunter's height and animated by the game.

Everything happens on your own PC. The mod only changes local model files,
never your save, the network, or anything other players download. Other
players keep seeing your real armor.

## What you need

1. Monster Hunter: World on Steam (base game or Iceborne).
2. **Stracker's Loader** ([Nexus Mods](https://www.nexusmods.com/monsterhunterworld/mods/1982)).
   Since Iceborne, the game only loads mods from `nativePC` when this loader
   is installed. Most modded setups already have it. Copy its files next to
   `MonsterHunterWorld.exe`. VRoid Hunter warns you if it can't find the loader.
3. Your avatar as a `.vrm` file. In VRoid Studio, use *Export → Export as VRM*.

## Install

1. Download `VRoidHunter.exe` from the latest
   [release](../../releases) (or from the *Actions* tab → latest run → *VRoidHunter* artifact).
2. Close the game and run `VRoidHunter.exe`. You can also drop your `.vrm` onto the exe.
3. Choose your `.vrm`. The game folder is detected from Steam; use *Browse* if it isn't.
4. Choose the armor set to replace. The default is the Leather set (`pl001_0000`).
   *Scan game* lists every set your game has.
5. Press **Install**. This takes about 10–60 seconds.
6. In game, equip that armor set on every slot. Its layered armor works too,
   so you can keep your real armor's skills. Make sure *Show headgear* is on.

Want a different avatar? Run the exe again and press **Install**; the old one
is replaced. To remove the mod, press **Uninstall**. If VRoid Hunter replaced
another mod's files, it restores them.

If you use a mod manager, **Export to folder…** writes a ready-made `nativePC`
folder instead of installing.

## What it does

* **Reads the game's own armor.** The exe opens your game's `chunkG*.bin`
  archives with the Oodle DLL that ships with the game. No game files are
  included in this download.
* **Fits the avatar to the hunter skeleton.** The avatar is rotated to face
  the right way, then scaled so its ankle-to-head height equals the hunter's.
  Each limb is then moved and stretched so its joints sit on the hunter's
  joints, arms included (the hunter's rest pose is an A-pose, not a T-pose).
  Because the mesh is bound to the hunter's own skeleton, the game's height
  slider scales your avatar too. The *Height* setting lets you scale it 50–150%.
* **Re-weights the mesh.** VRM humanoid bones are mapped to the hunter's bone
  IDs, including fingers and the forearm twist bone. Hair, skirt and accessory
  bones follow their nearest body bone.
* **Builds the files.** The body model (`.mod3`), its materials (`.mrl3`, cloned
  from the armor's own material so the shader parameters stay valid) and
  textures (`.tex`). The armor's other pieces (helm, arms, waist, legs) are
  replaced with invisible models so they don't overlap the avatar.
* **Hides the hunter's own face and hair** (optional, on by default) so they
  don't poke through the avatar's head.

## Limitations

* **Hair, skirts and accessories don't sway.** They move rigidly with the
  head or body bone they're attached to; VRM spring bones aren't converted to
  MHW physics.
* **No facial expressions.** The avatar keeps its default face.
* **Partly transparent parts.** Triangles that are fully see-through in the
  texture are removed, which handles most eyelashes, eye highlights and hair
  tips. Soft, semi-transparent edges render solid.
* **The whole armor set becomes the avatar.** Anyone else wearing the same set
  will also look like your avatar on your screen, so choose a set you rarely
  see.
* **"Hide face and hair" applies to every hunter on your screen.** Other
  players' hunters lose their face and hair too, unless their helmet covers it.
  Untick the option if that bothers you.
* **Not tested in the game yet.** The file formats come from community
  research and the converter is tested against synthetic data. Some armor sets
  may need a different choice. If one doesn't work, pick another set or open an issue.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| No change in game | Install Stracker's Loader. Equip the exact armor set you picked, on every slot. |
| "oo2core_8_win64.dll was not found" | The game folder is wrong. Choose the folder that contains `MonsterHunterWorld.exe`. |
| Avatar's head missing | Turn *Show headgear* on in the game options. |
| Game crashes on load | Press **Uninstall**, then try a different armor set. |

## Command line

```
VRoidHunter install  avatar.vrm [--game DIR] [--armor pl001_0000] [--keep-face] [--height 1.0] [--female-only]
VRoidHunter export   avatar.vrm OUTDIR [same options]
VRoidHunter uninstall [--game DIR]
VRoidHunter armors   [--game DIR]
```

`--extracted DIR` reads game files from a folder you already extracted
(e.g. with WorldChunkTool) instead of the chunk archives.

## Building from source

```
pip install -r requirements.txt pytest pyinstaller
python -m pytest
pyinstaller --onefile --windowed --name VRoidHunter --add-data "vroidhunter/data;vroidhunter/data" run_vroidhunter.py
```

GitHub Actions builds `VRoidHunter.exe` on every push (`.github/workflows/build.yml`)
and attaches it to a release when a `v*` tag is pushed.

## Credits

This tool stands on the MHW modding community's research. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details and licences.
