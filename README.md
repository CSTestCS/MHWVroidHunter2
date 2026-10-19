# VRoid Hunter

Play **Monster Hunter: World** (Steam) as your VRoid character. Give VRoid
Hunter any `.vrm` avatar (VRoid Studio, VRM 0.x or 1.0) and it turns one armor
set into your avatar: hair, skirts and accessories keep their physics, and
the avatar is sized to your hunter's height.

**Why an armor set?** In MHW your hunter's look *is* the armor they wear;
there's no separate character model to swap. Every model mod for the game
works this way. You pick one set (ideally one you own as layered armor),
and wearing it on all five slots makes your hunter your avatar. Nothing else
in the game changes. Other hunters and other armor look normal, and only your
PC sees the change.

## What you need

1. Monster Hunter: World on Steam (with Iceborne updates).
2. **Stracker's Loader** ([Nexus Mods](https://www.nexusmods.com/monsterhunterworld/mods/1982)),
   a one-time install that lets the game load mods at all. Copy its files next to
   `MonsterHunterWorld.exe`.
3. Your avatar as a `.vrm` file. In VRoid Studio, use *Export → Export as VRM*.

## Install

1. Download `VRoidHunter.exe` from the latest [release](../../releases)
   (or from the *Actions* tab → latest run → *VRoidHunter* artifact).
2. Close the game and run `VRoidHunter.exe`. You can also drop your `.vrm` onto it.
3. Choose your `.vrm`. The game folder is found through Steam.
4. Choose the armor set. VRoid Hunter checks every set in your game and marks
   the ones with a **full helmet**; those hide your hunter's own face under
   the avatar's head. Pick one you own as armor or layered armor.
5. Press **Install** (10–60 seconds).
6. In game, wear that set (or its layered armor) on all five slots and keep
   *Show headgear* on.

To use a different avatar, run it again and press **Install**. **Uninstall**
removes everything it added. **Export to folder…** writes a `nativePC` folder for mod managers.

### The one extra file (oo2core_8_win64.dll)

MHW stores its files compressed with a library called Oodle. The game only
includes an old copy that can't open its current files, so VRoid Hunter needs
the newer `oo2core_8_win64.dll` (or `oo2core_9`) to read your hunter's skeleton.
It looks for one in your other installed games first. If there isn't one, it
asks before downloading it from Warframe's public update server, which is where
it's freely hosted and where the community MHW tools get it too. The file is
checked against known checksums. You can also pick the file yourself.

## What it does

* **Keeps your avatar's proportions.** The avatar is turned to face the right
  way and scaled to your hunter's height. Its arms are rotated into the hunter's
  A-pose, and then the hunter's skeleton inside the armor files is moved to match
  the avatar's own joints. Long legs, a small head or short arms stay that way, and
  the game's animations bend the avatar at its own elbows and knees. The game's
  height slider still scales it. (Untick *Keep the VRoid model's own proportions*
  to stretch the avatar onto the hunter's build instead.)
* **Keeps the physics.** VRM spring bones (hair, skirts, ribbons, tails, chest)
  become MHW physics chains (`.ctc`). Their collision (`.ccl`) is built from the
  VRM's own colliders, or fitted to the avatar's body if it has none, so physics
  only reacts to the avatar and not to the replaced armor's shape.
* **Lines up the wrists.** Forearms are the one exception to keeping proportions:
  they take the hunter's forearm length and the arm's twist bones are spaced to
  match. The arms are then weighted to the twist and hand bones the same way the
  game's own armor is, so wrists don't twist or pinch in animations.
* **Hides the hunter's own neck.** The game still draws your hunter's face and neck
  under the helmet. Where it would poke out around a slimmer avatar neck, the
  avatar's neck is thickened just enough to cover it (at most 2x), and nothing
  else changes.
* **Matches the game's lighting.** Every part of the avatar uses a plain, non-metal
  player material cloned from the game, with the game's own wet settings, so it
  only looks wet when your hunter would (rain, water). Everything is matte by
  default, and colours are toned to the brightness range the game's lighting
  expects. (The game's own skin material isn't used: the game paints your
  hunter's skin texture over it.)
* **Splits the avatar like real armor.** The head and hair go in the helmet
  model, the rest in the body model. Armor pieces the avatar doesn't need (arms,
  waist, legs) are made invisible so they don't poke through.

## Limitations

* **Not tested in the game yet.** The file formats come from community research
  and the converter is tested on synthetic data and sample VRMs. Physics may need
  tuning. If something looks wrong, open an issue with a screenshot.
* **No facial expressions.** The avatar keeps its default face.
* **Partly transparent parts.** Fully see-through triangles are removed (eyelashes,
  eye highlights, hair tips), but soft semi-transparent edges render solid.
* **Physics bone limit.** The game allows about 90 extra physics bones per
  armor set. Very detailed hair is thinned out to fit, and every strand still moves.
* **Same set, same look.** Other players wearing the set you picked also look
  like your avatar on your screen, so pick one you rarely see.
* **Open-faced helmets.** With a set whose helmet doesn't cover the face, your
  hunter's own face may show through the avatar's head. Use a "full helmet" set.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| No change in game | Install Stracker's Loader. Wear the exact set you picked on all five slots. |
| Avatar has your hunter's head | Turn *Show headgear* on, and wear the helmet of the set. |
| Your hunter's face shows through | Pick a set marked "full helmet". |
| Game crashes on load | Press **Uninstall**, then try another set or untick physics. |
| Weapons or the slinger float away from the hands | Untick *Keep the VRoid model's own proportions*; very different arm lengths can confuse attachments. |

## Command line

```
VRoidHunter install  avatar.vrm [--game DIR] [--armor pl123_0000] [--no-physics] [--no-neck-cover] [--hunter-proportions] [--height 1.0] [--female-only]
                     [--oodle PATH | --download-oodle]
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
