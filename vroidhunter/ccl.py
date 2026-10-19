"""Writer for .ccl collision capsule files used by .ctc physics chains.

Layout from AsteriskAmpersand's CTC-MHW-Research: a 16-byte header and 64-byte
capsule records. Each capsule has two spheres; each sphere follows a base
skeleton bone (by bone function id) with an offset in that bone's space.
"""

import struct

HEADER_FMT = "<4s4Bii"
RECORD_FMT = "<4bhh8B4f4f12b4B"
RECORD_SIZE = struct.calcsize(RECORD_FMT)  # 64


def template_unknowns(data):
    """The unknown bytes of the first capsule of a real .ccl (8 front + 4 end)."""
    if not data or data[:4] != b"CCL\x00" or len(data) < 16 + RECORD_SIZE:
        return (0,) * 8, (0,) * 4
    vals = struct.unpack_from(RECORD_FMT, data, 16)
    return tuple(vals[6:14]), tuple(vals[34:38])


def write_ccl(capsules, unknowns=((0,) * 8, (0,) * 4)):
    """capsules: list of (func_a, offset_a, radius_a, func_b, offset_b, radius_b)."""
    front, end = unknowns
    out = bytearray(struct.pack(HEADER_FMT, b"CCL\x00", 24, 6, 8, 0, len(capsules), 0x40 * len(capsules)))
    for fa, oa, ra, fb, ob, rb in capsules:
        out += struct.pack(RECORD_FMT, 0, 0, 0, 0, int(fa), int(fb), *front,
                           *map(float, oa), float(ra), *map(float, ob), float(rb), *([0] * 12), *end)
    return bytes(out)


def read_ccl(data):
    n = struct.unpack_from("<i", data, 8)[0]
    out = []
    for i in range(n):
        v = struct.unpack_from(RECORD_FMT, data, 16 + RECORD_SIZE * i)
        out.append((v[4], v[14:17], v[17], v[5], v[18:21], v[21]))
    return out
