# Third-party notices

## MHWNoChunk (chunk archive reading)

`vroidhunter/data/chunk.key` is copied from, and `vroidhunter/chunk.py` is a
port of, [MHWNoChunk](https://github.com/JodoZT/MHWNoChunk) by Jodo, which in
turn builds on [WorldChunkTool](https://github.com/mhvuze/WorldChunkTool) by
MHVuze.

```
MIT License

Copyright (c) 2020 Jodo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## File format research

The `.mod3`, `.mrl3` and `.tex` code in this project was written from scratch
using documentation and research published by the MHW modding community:

* CrazyT – [MHW-Research](https://github.com/TheCrazyT/MHW-Research) 010 templates for mod3 and mrl3.
* AsteriskAmpersand – [Mod3-MHW-Importer](https://github.com/AsteriskAmpersand/Mod3-MHW-Importer)
  and [Material-Editing](https://github.com/AsteriskAmpersand/Material-Editing) research.
* Lukas Scone / PredatorCZ – original MT Framework model research.
* Jodo – [MHW Texture Converter](https://github.com/JodoZT/MHWTexConvertor) (tex header layout).
* Statyk, Karbon and contributors to the
  [MonsterHunterWorldModding wiki](https://github.com/Ezekial711/MonsterHunterWorldModding/wiki)
  – bone function tables, nativePC and texture documentation.

## Runtime dependencies

* [NumPy](https://numpy.org/) – BSD licence.
* [Pillow](https://python-pillow.org/) – HPND licence.
* Oodle (`oo2core_8_win64.dll`) is **not** distributed; the copy that ships
  with the game is loaded from the game folder.

Monster Hunter is a trademark of Capcom. This project is not affiliated with
Capcom, Pixiv (VRoid) or the VRM Consortium.
