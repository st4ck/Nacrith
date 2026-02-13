#!/usr/bin/env python3
"""Command-line interface for the neural text compressor."""

import argparse
import gzip
import sys
import time

from compressor import NeuralCompressor
from utils import format_size


def cmd_compress(args):
    binary_mode = False
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            text = f.read()
        raw = text.encode('utf-8')
    except UnicodeDecodeError:
        binary_mode = True
        with open(args.input, 'rb') as f:
            raw = f.read()

    original_size = len(raw)
    print(f"Original: {format_size(original_size)}")
    if binary_mode:
        print("Mode: binary (hybrid)")

    nc = NeuralCompressor(verbose=True)
    start = time.time()
    if binary_mode:
        compressed = nc.compress_bytes(raw)
    else:
        compressed = nc.compress(text)
    elapsed = time.time() - start

    with open(args.output, 'wb') as f:
        f.write(compressed)

    comp_size = len(compressed)
    ratio = comp_size / original_size if original_size > 0 else 0
    print(f"Compressed: {format_size(comp_size)}")
    print(f"Ratio: {ratio:.4f} ({100*ratio:.1f}%)")
    print(f"Time: {elapsed:.1f}s")


def cmd_decompress(args):
    with open(args.input, 'rb') as f:
        data = f.read()

    nc = NeuralCompressor(verbose=True)
    start = time.time()
    result = nc.decompress(data)
    elapsed = time.time() - start

    if isinstance(result, bytes):
        with open(args.output, 'wb') as f:
            f.write(result)
        print(f"Decompressed: {format_size(len(result))} (binary)")
    else:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"Decompressed: {format_size(len(result.encode('utf-8')))}")
    print(f"Time: {elapsed:.1f}s")


def cmd_benchmark(args):
    binary_mode = False
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            text = f.read()
        raw = text.encode('utf-8')
    except UnicodeDecodeError:
        binary_mode = True
        with open(args.input, 'rb') as f:
            raw = f.read()

    original_size = len(raw)
    print(f"Original: {format_size(original_size)}")
    if binary_mode:
        print("Mode: binary (hybrid)")

    # gzip
    gzip_data = gzip.compress(raw, compresslevel=9)
    gzip_size = len(gzip_data)
    print(f"gzip:     {format_size(gzip_size)} (ratio: {gzip_size/original_size:.4f})")

    # Neural
    nc = NeuralCompressor(verbose=True)
    start = time.time()
    if binary_mode:
        compressed = nc.compress_bytes(raw)
    else:
        compressed = nc.compress(text)
    elapsed = time.time() - start
    nc_size = len(compressed)
    print(f"Neural:   {format_size(nc_size)} (ratio: {nc_size/original_size:.4f})")
    print(f"Neural compression time: {elapsed:.1f}s")

    if nc_size < gzip_size:
        improvement = (1 - nc_size / gzip_size) * 100
        print(f"Neural is {improvement:.1f}% smaller than gzip")
    else:
        overhead = (nc_size / gzip_size - 1) * 100
        print(f"Neural is {overhead:.1f}% larger than gzip")


def main():
    parser = argparse.ArgumentParser(
        description="Neural text compressor using SmolLM2-135M + arithmetic coding"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_comp = sub.add_parser("compress", help="Compress a text file")
    p_comp.add_argument("input", help="Input text file")
    p_comp.add_argument("output", help="Output compressed file (.nc)")
    p_comp.set_defaults(func=cmd_compress)

    p_decomp = sub.add_parser("decompress", help="Decompress a .nc file")
    p_decomp.add_argument("input", help="Input compressed file (.nc)")
    p_decomp.add_argument("output", help="Output text file")
    p_decomp.set_defaults(func=cmd_decompress)

    p_bench = sub.add_parser("benchmark", help="Benchmark vs gzip")
    p_bench.add_argument("input", help="Input text file")
    p_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
