"""Read a TF1 tensor-bundle checkpoint (.index SSTable + .data shard) without TensorFlow.

Only what is needed here: uncompressed SSTable blocks, BundleEntryProto with
dtype/shape/shard/offset/size, little-endian float32/int32/int64 data.
"""
import struct
import numpy as np


def _varint(buf, pos):
    result = shift = 0
    while True:
        b = buf[pos]; pos += 1
        result |= (b & 0x7f) << shift
        if b < 0x80: return result, pos
        shift += 7


def _block_entries(buf, offset, size):
    block = buf[offset:offset + size]
    compression = buf[offset + size]
    if compression != 0: raise ValueError('compressed SSTable block not supported')
    restarts = struct.unpack('<I', block[-4:])[0]
    end = len(block) - 4 - 4 * restarts
    pos, key = 0, b''
    while pos < end:
        shared, pos = _varint(block, pos)
        unshared, pos = _varint(block, pos)
        vlen, pos = _varint(block, pos)
        key = key[:shared] + block[pos:pos + unshared]; pos += unshared
        value = block[pos:pos + vlen]; pos += vlen
        yield key, value


def _handle(value, pos=0):
    off, pos = _varint(value, pos)
    size, pos = _varint(value, pos)
    return off, size, pos


def _proto(value):
    """Decode a protobuf message into {field: [values]} (varint, fixed32, len-delimited)."""
    fields, pos = {}, 0
    while pos < len(value):
        tag, pos = _varint(value, pos)
        number, wire = tag >> 3, tag & 7
        if wire == 0: v, pos = _varint(value, pos)
        elif wire == 2:
            n, pos = _varint(value, pos); v = value[pos:pos + n]; pos += n
        elif wire == 5: v = struct.unpack('<I', value[pos:pos + 4])[0]; pos += 4
        elif wire == 1: v = struct.unpack('<Q', value[pos:pos + 8])[0]; pos += 8
        else: raise ValueError('unsupported wire type %d' % wire)
        fields.setdefault(number, []).append(v)
    return fields


DTYPES = {1: np.float32, 3: np.int32, 9: np.int64, 2: np.float64, 10: np.bool_}


def read(prefix):
    index = open(prefix + '.index', 'rb').read()
    data = open(prefix + '.data-00000-of-00001', 'rb').read()
    footer = index[-48:]
    _, _, pos = _handle(footer)          # metaindex handle
    ioff, isize, _ = _handle(footer, pos)  # index handle
    tensors = {}
    for _, handle in _block_entries(index, ioff, isize):
        off, size, _ = _handle(handle)
        for key, value in _block_entries(index, off, size):
            if key == b'': continue  # bundle header
            entry = _proto(value)
            dtype = entry.get(1, [0])[0]
            if dtype not in DTYPES: continue  # e.g. strings
            dims = []
            for shape in entry.get(2, []):
                for dim in _proto(shape).get(2, []):
                    dims.append(_proto(dim).get(1, [0])[0])
            offset = entry.get(4, [0])[0]; nbytes = entry.get(5, [0])[0]
            array = np.frombuffer(data[offset:offset + nbytes], dtype=DTYPES[dtype]).reshape(dims)
            tensors[key.decode()] = array.copy()
    return tensors


if __name__ == '__main__':
    import sys
    t = read(sys.argv[1])
    for k in sorted(t):
        if k.startswith(('critic', 'abstract_actor')) and 'Adam' not in k:
            print(k, t[k].shape, t[k].dtype)
