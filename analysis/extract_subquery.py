#!/usr/bin/env python3
"""
extract_subquery.py
Extracts first N queries from a DiskANN binary query file.
Format: [uint32 npts][uint32 ndims][float32 * npts * ndims]
"""
import sys
import struct
import numpy as np

def extract_subquery(src_path: str, dst_path: str, n: int):
    with open(src_path, 'rb') as f:
        npts = struct.unpack('<I', f.read(4))[0]
        ndims = struct.unpack('<I', f.read(4))[0]
        print(f"Source: {npts} pts × {ndims} dims")
        n = min(n, npts)
        data = np.frombuffer(f.read(n * ndims * 4), dtype=np.float32)
        data = data.reshape(n, ndims)

    with open(dst_path, 'wb') as f:
        f.write(struct.pack('<I', n))
        f.write(struct.pack('<I', ndims))
        f.write(data.tobytes())

    print(f"Saved {n} queries → {dst_path}")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: extract_subquery.py <src.bin> <dst.bin> <n>")
        sys.exit(1)
    extract_subquery(sys.argv[1], sys.argv[2], int(sys.argv[3]))
