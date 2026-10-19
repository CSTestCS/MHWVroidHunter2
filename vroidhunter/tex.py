"""Writer for Monster Hunter: World .tex textures.

Header layout mirrors Jodo's MHW Texture Converter (dds -> tex direction).
Supports uncompressed RGBA8 (linear or sRGB) and BC1 (linear or sRGB), with a
full mip chain generated here.
"""

import struct

import numpy as np

FMT_RGBA8 = 7
FMT_RGBA8_SRGB = 9
FMT_BC1 = 22
FMT_BC1_SRGB = 23

_COMPRESSED = {FMT_BC1, FMT_BC1_SRGB}


def next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def fit_size(width, height, max_size=2048, min_size=8):
    """Power-of-two size no larger than max_size, keeping aspect roughly."""
    w = min(max(next_pow2(width), min_size), max_size)
    h = min(max(next_pow2(height), min_size), max_size)
    # Avoid upscaling a lot: if next_pow2 doubled a dimension well past the source, step down.
    if w > width * 1.5 and w > min_size:
        w //= 2
    if h > height * 1.5 and h > min_size:
        h //= 2
    return max(w, min_size), max(h, min_size)


def mip_chain(rgba, min_dim=4):
    """Box-filtered mip levels down to min_dim (inclusive)."""
    levels = [rgba]
    cur = rgba.astype(np.float32)
    while cur.shape[0] // 2 >= min_dim and cur.shape[1] // 2 >= min_dim:
        cur = (cur[0::2, 0::2] + cur[1::2, 0::2] + cur[0::2, 1::2] + cur[1::2, 1::2]) * 0.25
        levels.append(np.clip(np.round(cur), 0, 255).astype(np.uint8))
    return levels


def _rgb565(c):
    c = np.clip(np.round(c), 0, 255).astype(np.int32)
    return ((c[..., 0] * 31 + 127) // 255 << 11) | ((c[..., 1] * 63 + 127) // 255 << 5) | ((c[..., 2] * 31 + 127) // 255)


def _unpack565(v):
    r = ((v >> 11) & 31) * 255.0 / 31
    g = ((v >> 5) & 63) * 255.0 / 63
    b = (v & 31) * 255.0 / 31
    return np.stack([r, g, b], axis=-1)


def encode_bc1(rgba):
    """Vectorized BC1 (DXT1) encoder, opaque 4-color mode."""
    h, w = rgba.shape[:2]
    blocks = rgba[:, :, :3].astype(np.float32).reshape(h // 4, 4, w // 4, 4, 3).transpose(0, 2, 1, 3, 4)
    blocks = blocks.reshape(-1, 16, 3)
    mean = blocks.mean(axis=1, keepdims=True)
    centered = blocks - mean
    # Principal axis by a few power iterations on the covariance.
    cov = np.einsum("bni,bnj->bij", centered, centered)
    axis = np.ones((len(blocks), 3), dtype=np.float32)
    for _ in range(4):
        axis = np.einsum("bij,bj->bi", cov, axis)
        norm = np.linalg.norm(axis, axis=1, keepdims=True)
        norm[norm == 0] = 1
        axis = axis / norm
    proj = np.einsum("bni,bi->bn", centered, axis)
    c0 = mean[:, 0] + axis * proj.max(axis=1, keepdims=True)
    c1 = mean[:, 0] + axis * proj.min(axis=1, keepdims=True)
    e0 = _rgb565(c0)
    e1 = _rgb565(c1)
    swap = e0 < e1
    e0, e1 = np.where(swap, e1, e0), np.where(swap, e0, e1)
    same = e0 == e1
    # Equal endpoints would switch to 3-colour mode; nudge to keep 4-colour mode.
    bump = same & (e0 < 0xFFFF)
    e0 = np.where(bump, e0 + 1, e0)
    e1 = np.where(same & ~bump, e1 - 1, e1)
    p0, p1 = _unpack565(e0), _unpack565(e1)
    palette = np.stack([p0, p1, (2 * p0 + p1) / 3, (p0 + 2 * p1) / 3], axis=1)  # (b,4,3)
    dist = ((blocks[:, :, None, :] - palette[:, None, :, :]) ** 2).sum(axis=-1)  # (b,16,4)
    idx = dist.argmin(axis=-1).astype(np.uint32)
    shifts = (np.arange(16, dtype=np.uint32) * 2)
    bits = (idx << shifts).sum(axis=1).astype("<u4")
    out = np.zeros((len(blocks), 8), dtype=np.uint8)
    out[:, 0:2] = e0.astype("<u2").view(np.uint8).reshape(-1, 2)
    out[:, 2:4] = e1.astype("<u2").view(np.uint8).reshape(-1, 2)
    out[:, 4:8] = bits.view(np.uint8).reshape(-1, 4)
    return out.tobytes()


def write_tex(rgba, fmt):
    """rgba: (h, w, 4) uint8 array with power-of-two sides >= 4."""
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    height, width = rgba.shape[:2]
    if width & (width - 1) or height & (height - 1) or width < 4 or height < 4:
        raise ValueError("texture size must be a power of two >= 4")
    levels = mip_chain(rgba)
    if fmt in _COMPRESSED:
        payloads = [encode_bc1(level) for level in levels]
    else:
        payloads = [level.tobytes() for level in levels]
    mip_count = len(levels)

    out = bytearray()
    out += bytes.fromhex("5445580010000000000000000000000002000000")
    out += struct.pack("<IIIII", mip_count, width, height, 1, fmt)
    out += bytes.fromhex("01000000000000000000000000000000FFFFFFFF0000000000000000")
    out += struct.pack("<I", 0)              # 1 only for BC6/BC7
    out += b"\x00" * 16
    out += b"\xff" * 32
    full_width = fmt not in _COMPRESSED
    pitch = width if full_width else width // 2
    out += struct.pack("<I", width)
    for i in range(3):
        out += struct.pack("<HH", pitch & 0xFFFF, width & 0xFFFF)
        out += b"\x00" * (8 if i < 2 else 32)
    assert len(out) == 0xB8, hex(len(out))
    offset = 0xB8 + 8 * mip_count
    for payload in payloads:
        out += struct.pack("<Q", offset)
        offset += len(payload)
    for payload in payloads:
        out += payload
    return bytes(out)


def read_tex_header(data):
    """Minimal parser used by tests: returns (mip_count, width, height, fmt, offsets)."""
    if data[:4] != b"TEX\x00":
        raise ValueError("not a tex file")
    mip_count, width, height, _depth, fmt = struct.unpack_from("<IIIII", data, 0x14)
    offsets = struct.unpack_from("<%dQ" % mip_count, data, 0xB8)
    return mip_count, width, height, fmt, offsets
