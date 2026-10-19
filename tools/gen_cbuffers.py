"""Generate vroidhunter/data/cbuffers.json from CrazyT's MHW-Research shader dump.

Usage: python tools/gen_cbuffers.py path/to/mhwib_structures_generated.bt
Source: https://github.com/TheCrazyT/MHW-Research (MIT licence).
Keeps the material constant buffers: their size and the offset/type of each field.
"""

import json
import re
import sys
import zlib

FIELD = re.compile(r"(\w+)\s+(\w+)(\[\d+\])?;\s*//\s*Offset:\s*(\d+)\s+Size:\s*(\d+)")


def main(path, out):
    text = open(path, encoding="utf-8", errors="replace").read()
    table = {}
    for m in re.finditer(r"struct (CB\w+) \{(.*?)\};", text, re.S):
        name, body = m.group(1), m.group(2)
        if not (name.startswith("CBMhMaterial") or name.startswith("CBMaterial")):
            continue
        fields, size = {}, 0
        for f in FIELD.finditer(body):
            ftype, fname, _arr, off, sz = f.groups()
            if fname.startswith("align"):
                continue
            fields[fname] = [int(off), ftype]
            size = max(size, int(off) + int(sz))
        size = (size + 15) // 16 * 16
        key = ((~zlib.crc32(name.encode())) & 0xFFFFF)
        table[str(key)] = {"name": name, "size": size, "fields": fields}
    with open(out, "w") as fh:
        json.dump(table, fh, separators=(",", ":"), sort_keys=True)
    print("%d constant buffers" % len(table))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "vroidhunter/data/cbuffers.json")
