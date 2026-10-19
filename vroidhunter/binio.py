"""Small binary helpers shared by the file format modules."""

import struct
import zlib


def jamcrc(text):
    """CRC-32/JAMCRC, the hash MT Framework uses for names."""
    if isinstance(text, str):
        text = text.encode("ascii")
    return (~zlib.crc32(text)) & 0xFFFFFFFF


def align(offset, grid=16):
    rem = offset % grid
    return offset + (grid - rem if rem else 0)


def pad_to(buf, offset):
    """Pad a bytearray with zeros until it is `offset` bytes long."""
    if len(buf) > offset:
        raise ValueError("buffer already past offset %d (%d)" % (offset, len(buf)))
    buf.extend(b"\x00" * (offset - len(buf)))


class Reader:
    def __init__(self, data, offset=0):
        self.data = data
        self.pos = offset

    def seek(self, pos):
        self.pos = pos

    def read(self, n):
        if self.pos + n > len(self.data):
            raise ValueError("read past end of data (%d+%d > %d)" % (self.pos, n, len(self.data)))
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def unpack(self, fmt):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read(size))

    def u8(self):
        return self.unpack("<B")[0]

    def u16(self):
        return self.unpack("<H")[0]

    def i32(self):
        return self.unpack("<i")[0]

    def u32(self):
        return self.unpack("<I")[0]

    def u64(self):
        return self.unpack("<Q")[0]
