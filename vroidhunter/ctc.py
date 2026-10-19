"""Writer for .ctc physics chain files (MHW "jiggle bones").

Layout from AsteriskAmpersand's CTC research and the MHW modding wiki:
an 80-byte header, one 80-byte record per chain, then one record per node
(112 bytes in Iceborne; older files used 96). The tuning values are copied
from a real .ctc in the player's game (preferably a hair one), so only the
chain lengths, bone function ids and node frames are written by us.
"""

import struct
from dataclasses import dataclass

HEADER_SIZE = 80
CHAIN_SIZE = 80
NODE_MATRIX = 0          # 16 floats
NODE_FIXED_END = 66      # byte: 1 = chain anchor
NODE_FUNCTION = 72       # int32 bone function id


@dataclass
class CtcTemplate:
    header: bytes
    chain: bytes
    first_node: bytes
    node: bytes


def parse_template(data):
    """Pick the longest chain of a real .ctc file as the template."""
    if data[:4] != b"CTC\x00" or len(data) < HEADER_SIZE:
        return None
    n_chains, n_nodes = struct.unpack_from("<ii", data, 16)
    if n_chains <= 0 or n_nodes <= 0:
        return None
    rest = len(data) - HEADER_SIZE - CHAIN_SIZE * n_chains
    if rest <= 0 or rest % n_nodes:
        return None
    node_size = rest // n_nodes
    if node_size not in (96, 112):
        return None
    nodes_at = HEADER_SIZE + CHAIN_SIZE * n_chains
    best, start, best_start = None, 0, 0
    for i in range(n_chains):
        rec = data[HEADER_SIZE + CHAIN_SIZE * i:HEADER_SIZE + CHAIN_SIZE * (i + 1)]
        length = struct.unpack_from("<i", rec, 0)[0]
        if length >= 2 and (best is None or length > struct.unpack_from("<i", best, 0)[0]):
            best, best_start = rec, start
        start += max(length, 0)
    if best is None:
        return None
    first = data[nodes_at + node_size * best_start:nodes_at + node_size * (best_start + 1)]
    second = data[nodes_at + node_size * (best_start + 1):nodes_at + node_size * (best_start + 2)]
    return CtcTemplate(data[:HEADER_SIZE], best, first, second)


def default_template():
    """Fallback values (AsteriskAmpersand's CTC plugin defaults, Iceborne layout)."""
    header = struct.pack("<4s3iiii9f2f6b2b", b"CTC\x00", 28, 0, 1000, 0, 0, 64,
                         1 / 60, 1.0, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0, 0.3, 0.5, 0.2,
                         1, 1, 1, 1, 1, 1, 0, 0)
    chain = struct.pack("<iBB2b4b4b4b12b11fhH", 0, 4, 49, 0, 0, -1, -1, -1, -1, 1, 0, 0, 0, 1, 0, 0, 0,
                        *([-51] * 12), 0, -980, 0, 0, 0.25, 1, 0.25, 100, 0, 0.1, 1.0, -1, 0xCDCD)
    ident = [1.0 if i == j else 0.0 for j in range(4) for i in range(4)]
    node = struct.pack("<16f2bB5BiB3bf3ff12b", *ident, 0, 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0,
                       1.0, 1, 1, 1, 1.0, *([-51] * 12))
    first = bytearray(node)
    first[NODE_FIXED_END] = 1
    return CtcTemplate(header, chain, bytes(first), node)


NODE_RADIUS = 80         # float, collision radius (112-byte Iceborne nodes)
CHAIN_COLLISION = 4      # byte: 4 = collide with the .ccl capsules, 0 = no collision


def _frame(direction):
    """Node frame with its X axis along the chain (as the community CTC tools build it)."""
    import numpy as np
    d = np.asarray(direction, dtype=np.float64)
    n = np.linalg.norm(d)
    m = np.eye(4)
    if n < 1e-9:
        return m
    x = d / n
    helper = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y = np.cross(helper, x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    m[:3, 0], m[:3, 1], m[:3, 2] = x, y, z
    return m


def _node(template_node, function, direction=None, radius=None):
    import numpy as np
    rec = bytearray(template_node)
    m = _frame(direction) if direction is not None else np.eye(4)
    struct.pack_into("<16f", rec, NODE_MATRIX, *m.T.reshape(16))   # stored column by column
    struct.pack_into("<i", rec, NODE_FUNCTION, function)
    if radius is not None and len(rec) >= NODE_RADIUS + 4 and len(rec) == 112:
        struct.pack_into("<f", rec, NODE_RADIUS, radius)
    return bytes(rec)


def write_ctc(template, chains, collide=True):
    """chains: lists of bone function ids (anchor first), or dicts with keys
    funcs, dirs (direction to the next node, per node) and radius."""
    norm = []
    for c in chains:
        c = c if isinstance(c, dict) else {"funcs": list(c)}
        if len(c["funcs"]) >= 2:
            norm.append(c)
    header = bytearray(template.header)
    struct.pack_into("<ii", header, 16, len(norm), sum(len(c["funcs"]) for c in norm))
    out = bytearray(header)
    for c in norm:
        rec = bytearray(template.chain)
        struct.pack_into("<i", rec, 0, len(c["funcs"]))
        rec[CHAIN_COLLISION] = 4 if collide else 0
        out += rec
    for c in norm:
        dirs = c.get("dirs") or [None] * len(c["funcs"])
        radius = c.get("radius")
        for i, f in enumerate(c["funcs"]):
            out += _node(template.first_node if i == 0 else template.node, f, dirs[i], radius)
    return bytes(out)


def read_chains(data):
    """Parse back (for tests): list of (fixed flags, function ids) per chain."""
    n_chains, n_nodes = struct.unpack_from("<ii", data, 16)
    node_size = (len(data) - HEADER_SIZE - CHAIN_SIZE * n_chains) // n_nodes
    pos = HEADER_SIZE + CHAIN_SIZE * n_chains
    out = []
    for i in range(n_chains):
        length = struct.unpack_from("<i", data, HEADER_SIZE + CHAIN_SIZE * i)[0]
        fixed, funcs = [], []
        for _ in range(length):
            fixed.append(data[pos + NODE_FIXED_END])
            funcs.append(struct.unpack_from("<i", data, pos + NODE_FUNCTION)[0])
            pos += node_size
        out.append((fixed, funcs))
    return out
