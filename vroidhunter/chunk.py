"""Read files straight out of Monster Hunter: World's chunkG*.bin archives.

Port of the chunk logic in JodoZT's MHWNoChunk (MIT licence, see
THIRD_PARTY_NOTICES.md), which is based on mhvuze's WorldChunkTool.
Decompression uses oo2core_8_win64.dll, which ships with the game.
"""

import ctypes
import os
import re
import struct
import sys

SUBCHUNK = 0x40000
CHUNK_MAGIC = 0x00504D43

CHUNK_KEYS = [bytes.fromhex(h) for h in (
    "ac76cb97ec7500133a81038e7a82c80a", "6bb5c956e44d00bc305233cfbfaafa25",
    "8f021dccb0f2787206fbdee2390bbb5c", "da5c1e531d8359157875bd63bfaafa25",
    "1f6d31c883dd716d7e8f598ce23f1929", "4bb0de04e4e0856980ccb2942f9ce9f9",
    "8bce54dc4c11139a7875bd63bfaafa25", "ec13345966ce7312440089a2ceddcee9",
    "e4662c709c753a039a2c0f5ae23f1929", "a492fc9033949c15a033ac223735cca7",
    "25099c1f911a26e5ce9172f07a82c80a", "d1d29d7446d4fdf1a033ac223735cca7",
    "7eb268373b5d361ed6d313e2933c4dcb", "a1c7d2ea661895ac7875bd63bfaafa25",
    "82a43b2108797c6a440089a2ceddcee9", "41d055b3dd6015167e8f598ce23f1929",
)]


def _data_path(name):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(base, "data", name)
    if os.path.exists(candidate):
        return candidate
    return os.path.join(base, "vroidhunter", "data", name)


_KEY_PATTERN = None


def key_pattern():
    global _KEY_PATTERN
    if _KEY_PATTERN is None:
        with open(_data_path("chunk.key"), "rb") as f:
            data = f.read()
        start, end = struct.unpack_from("<ii", data, 0)
        # Entry for key iterator k (start..end) is the sub-chunk index k-1.
        _KEY_PATTERN = (start - 1, data[8:8 + (end - start + 1)])
    return _KEY_PATTERN


def chunk_key(index):
    first, pattern = key_pattern()
    pos = index - first
    key = pattern[pos] if 0 <= pos < len(pattern) else 0
    if key > 0xF:
        key = 0
    return CHUNK_KEYS[key]


def _xor(data, key):
    # Fast XOR with a repeating 16-byte key.
    n = len(data)
    reps = (n + len(key) - 1) // len(key)
    stream = (key * reps)[:n]
    return (int.from_bytes(data, "little") ^ int.from_bytes(stream, "little")).to_bytes(n, "little")


class Oodle:
    def __init__(self, dll_path):
        if os.name == "nt":
            self.lib = ctypes.WinDLL(dll_path)
        else:
            self.lib = ctypes.CDLL(dll_path)
        fn = self.lib.OodleLZ_Decompress
        fn.restype = ctypes.c_int64
        fn.argtypes = [ctypes.c_char_p, ctypes.c_int64, ctypes.c_char_p, ctypes.c_int64,
                       ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint32,
                       ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                       ctypes.c_uint32, ctypes.c_uint32]
        self.fn = fn

    def decompress(self, data, out_size):
        out = ctypes.create_string_buffer(out_size)
        got = self.fn(data, len(data), out, out_size, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3)
        if got <= 0:
            raise IOError("Oodle decompression failed")
        return out.raw[:got]


class ChunkArchive:
    def __init__(self, path, oodle):
        self.path = path
        self.oodle = oodle
        self.f = open(path, "rb")
        magic, count = struct.unpack("<ii", self.f.read(8))
        if magic != CHUNK_MAGIC:
            raise ValueError("%s is not a chunk archive" % path)
        table = self.f.read(8 * count)
        self.sub = []
        for i in range(count):
            raw = bytearray(table[8 * i:8 * i + 3])
            raw[0] = raw[0] >> 4
            value = int.from_bytes(bytes(raw), "little")
            size = (value >> 4) + (value & 0xF)
            offset = int.from_bytes(table[8 * i + 3:8 * i + 8], "little")
            self.sub.append((offset, size))
        self._cache = {}
        self.files = {}
        self._read_file_table()

    def close(self):
        self.f.close()

    def subchunk(self, index):
        if index in self._cache:
            return self._cache[index]
        offset, size = self.sub[index]
        self.f.seek(offset)
        if size:
            data = self.oodle.decompress(self.f.read(size), SUBCHUNK)
        else:
            data = self.f.read(SUBCHUNK)
        data = _xor(data, chunk_key(index))
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[index] = data
        return data

    def read(self, offset, length):
        out = bytearray()
        while length > 0:
            index, pos = divmod(offset, SUBCHUNK)
            block = self.subchunk(index)
            take = min(length, len(block) - pos)
            if take <= 0:
                raise IOError("read past end of chunk archive")
            out += block[pos:pos + take]
            offset += take
            length -= take
        return bytes(out)

    def _read_file_table(self):
        head = self.read(0, 0x14)
        parents, _children = struct.unpack_from("<ii", head, 0x0C)
        pos = 0x100
        for _ in range(parents):
            # name[0x3C], size i64, offset i64, type i32, child count i32
            entry = self.read(pos, 0x54)
            pos += 0x54
            child_count = struct.unpack_from("<i", entry, 0x50)[0]
            blob = self.read(pos, 0xB8 * child_count)
            pos += 0xB8 * child_count
            for j in range(child_count):
                e = blob[0xB8 * j:0xB8 * (j + 1)]
                size, offset, etype = struct.unpack_from("<qqi", e, 0xA0)
                if etype not in (0, 2):
                    continue
                name = e[:0xA0].split(b"\x00", 1)[0].decode("ascii", "replace")
                name = name.strip("\\").replace("/", "\\")
                if name:
                    self.files[name.lower()] = (name, offset, size)

    def get(self, key):
        name, offset, size = self.files[key]
        return self.read(offset, size)


def _chunk_number(path):
    m = re.search(r"chunkG(\d+)\.bin$", path, re.IGNORECASE)
    return int(m.group(1)) if m else -1


class GameFiles:
    """Union view over every chunk archive; later chunks override earlier ones."""

    def __init__(self, game_dir, log=print):
        self.game_dir = game_dir
        dll = os.path.join(game_dir, "oo2core_8_win64.dll")
        if not os.path.exists(dll):
            raise FileNotFoundError("oo2core_8_win64.dll was not found in the game folder")
        self.oodle = Oodle(dll)
        chunk_dir = os.path.join(game_dir, "chunk")
        paths = []
        if os.path.isdir(chunk_dir):
            paths = [os.path.join(chunk_dir, p) for p in os.listdir(chunk_dir) if _chunk_number(p) >= 0]
        if not paths:
            raise FileNotFoundError("no chunkG*.bin files found in %s" % chunk_dir)
        paths.sort(key=_chunk_number)
        self.archives = []
        self.index = {}
        for p in paths:
            log("Reading file list from %s" % os.path.basename(p))
            arc = ChunkArchive(p, self.oodle)
            self.archives.append(arc)
            for key in arc.files:
                self.index[key] = arc

    def exists(self, path):
        return path.lower().replace("/", "\\").strip("\\") in self.index

    def read(self, path):
        key = path.lower().replace("/", "\\").strip("\\")
        return self.index[key].get(key)

    def listdir(self, prefix):
        """All file paths (original case) starting with prefix."""
        p = prefix.lower().replace("/", "\\").strip("\\")
        out = []
        for key, arc in self.index.items():
            if key.startswith(p):
                out.append(arc.files[key][0])
        return sorted(out)

    def close(self):
        for a in self.archives:
            a.close()


class FolderFiles:
    """Same interface as GameFiles, over a folder of already-extracted game files."""

    def __init__(self, root):
        self.root = root
        self.index = {}
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                full = os.path.join(dirpath, f)
                rel = os.path.relpath(full, root).replace(os.sep, "\\")
                self.index[rel.lower()] = (rel, full)

    def exists(self, path):
        return path.lower().replace("/", "\\").strip("\\") in self.index

    def read(self, path):
        with open(self.index[path.lower().replace("/", "\\").strip("\\")][1], "rb") as f:
            return f.read()

    def listdir(self, prefix):
        p = prefix.lower().replace("/", "\\").strip("\\")
        return sorted(v[0] for k, v in self.index.items() if k.startswith(p))

    def close(self):
        pass
