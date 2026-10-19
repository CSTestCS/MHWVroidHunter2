import struct

from vroidhunter.chunk import SUBCHUNK, ChunkArchive, _xor, chunk_key


class NoOodle:
    def decompress(self, data, size):
        raise AssertionError("raw sub-chunks only in this test")


def _make_archive(path, files):
    """Build a chunk archive with uncompressed (size 0) sub-chunks."""
    names = list(files)
    stream = bytearray(SUBCHUNK)  # table lives in the first sub-chunk(s)
    struct.pack_into("<ii", stream, 0x0C, 1, len(names))
    pos = 0x100
    entry = bytearray(0x54)
    entry[:4] = b"\\\x00\x00\x00"
    struct.pack_into("<qqii", entry, 0x3C, 0, 0, 1, len(names))
    stream[pos:pos + 0x54] = entry
    pos += 0x54
    data_at = SUBCHUNK - 100  # first file straddles a sub-chunk boundary
    blobs = bytearray()
    for name in names:
        e = bytearray(0xB8)
        raw = ("\\" + name).encode()
        e[:len(raw)] = raw
        struct.pack_into("<qqii", e, 0xA0, len(files[name]), data_at + len(blobs), 0, 0)
        stream[pos:pos + 0xB8] = e
        pos += 0xB8
        blobs += files[name]
    stream = stream[:data_at] + blobs
    while len(stream) % SUBCHUNK:
        stream.append(0)
    count = len(stream) // SUBCHUNK
    header = struct.pack("<ii", 0x00504D43, count)
    table_size = 8 + 8 * count
    out = bytearray(header)
    for i in range(count):
        out += bytes(3) + (table_size + i * SUBCHUNK).to_bytes(5, "little")
    for i in range(count):
        out += _xor(bytes(stream[i * SUBCHUNK:(i + 1) * SUBCHUNK]), chunk_key(i))
    with open(path, "wb") as f:
        f.write(out)


def test_chunk_reads_files(tmp_path):
    files = {"pl\\f_equip\\pl001_0000\\body\\mod\\f_body001.mod3": bytes(range(256)) * 10,
             "pl\\hair\\hair000\\mod\\hair000.mrl3": b"MRL\x00hello"}
    p = tmp_path / "chunkG0.bin"
    _make_archive(str(p), files)
    arc = ChunkArchive(str(p), NoOodle())
    for name, data in files.items():
        assert arc.get(name.lower()) == data
    arc.close()


def test_key_pattern_loaded():
    assert len(chunk_key(0)) == 16
    assert len(chunk_key(123456)) == 16
