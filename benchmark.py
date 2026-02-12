#!/usr/bin/env python3
"""Benchmark neural compressor vs gzip, xz, zip on various text sizes."""

import gzip
import lzma
import time
import zipfile
import io
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from compressor import NeuralCompressor
from utils import format_size

SAMPLES = {
    "tiny": (
        "The quick brown fox jumps over the lazy dog."
    ),
    "small": (
        "Natural language processing has seen remarkable advances in recent years. "
        "Large language models, trained on vast corpora of text data, have demonstrated "
        "an impressive ability to understand and generate human language. These models "
        "capture statistical patterns in text — from simple word co-occurrences to complex "
        "semantic relationships — making them powerful tools for prediction and generation."
    ),
    "medium": (
        "The history of data compression is deeply intertwined with information theory. "
        "Claude Shannon, in his landmark 1948 paper 'A Mathematical Theory of Communication', "
        "established the theoretical foundations for data compression by defining entropy as "
        "the fundamental limit of lossless compression. Shannon showed that the entropy of a "
        "source measures the average amount of information per symbol, and no lossless "
        "compression scheme can compress data below this limit on average.\n\n"
        "Since Shannon's work, numerous compression algorithms have been developed. "
        "Huffman coding, invented in 1952, assigns variable-length codes to symbols based "
        "on their frequencies. The Lempel-Ziv family of algorithms (LZ77, LZ78, LZW) "
        "exploits repeated patterns in data by replacing them with references to earlier "
        "occurrences. Arithmetic coding, developed in the 1970s, can achieve compression "
        "ratios very close to the theoretical entropy limit by encoding entire messages "
        "as single numbers in the interval [0, 1).\n\n"
        "Modern compression tools like gzip use a combination of LZ77 and Huffman coding. "
        "More advanced compressors like xz use the LZMA algorithm, which employs a "
        "sophisticated dictionary-based approach with range coding. These traditional "
        "methods work well for general data but are limited in their ability to model "
        "complex dependencies in natural language text."
    ),
    "large": None,  # Will be generated
}

# Generate large sample by repeating and varying medium content
SAMPLES["large"] = (
    SAMPLES["medium"] + "\n\n"
    "The connection between prediction and compression is fundamental. A good predictor "
    "can be turned into a good compressor, and vice versa. This insight has led researchers "
    "to explore the use of neural language models as the prediction engine in compression "
    "systems. The idea is simple yet powerful: if a model can predict the next token with "
    "high confidence, very few bits are needed to encode that token.\n\n"
    "Arithmetic coding provides the ideal framework for turning predictions into compressed "
    "bits. Given a probability distribution over possible next tokens, the arithmetic coder "
    "narrows an interval proportionally to each token's probability. Tokens predicted with "
    "high confidence consume very few bits — a token predicted with 99%% probability costs "
    "only about 0.014 bits. Conversely, surprising tokens require more bits to encode.\n\n"
    "Recent work has demonstrated that neural language models, even relatively small ones, "
    "can outperform traditional compression algorithms on natural language text. This is "
    "because language models capture long-range dependencies and semantic structure that "
    "dictionary-based methods cannot. A language model understands that 'The President of "
    "the United States' is a likely phrase, while gzip needs to have seen that exact byte "
    "sequence recently in its sliding window.\n\n"
    "The trade-off is speed. Traditional compressors like gzip process data at hundreds of "
    "megabytes per second. Neural compressors require a forward pass through the model for "
    "each token, making them orders of magnitude slower. However, with small, efficient "
    "models and hardware acceleration, the gap is narrowing. For archival storage where "
    "compression ratio matters more than speed, neural compression offers a compelling "
    "advantage over traditional methods."
)


def compress_gzip(data: bytes) -> bytes:
    return gzip.compress(data, compresslevel=9)


def compress_xz(data: bytes) -> bytes:
    return lzma.compress(data, preset=9)


def compress_zip(data: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr('data.txt', data)
    return buf.getvalue()


def main():
    print("Loading neural compressor...")
    nc = NeuralCompressor(verbose=True)
    print()

    results = []

    for name, text in SAMPLES.items():
        raw = text.encode('utf-8')
        original_size = len(raw)
        print(f"{'='*70}")
        print(f"Sample: {name} ({format_size(original_size)})")
        print(f"{'='*70}")

        row = {"name": name, "original": original_size}

        # gzip
        gz = compress_gzip(raw)
        row["gzip"] = len(gz)
        print(f"  gzip:   {format_size(len(gz)):>10s}  ratio: {len(gz)/original_size:.4f}  ({100*len(gz)/original_size:.1f}%)")

        # xz
        xz = compress_xz(raw)
        row["xz"] = len(xz)
        print(f"  xz:     {format_size(len(xz)):>10s}  ratio: {len(xz)/original_size:.4f}  ({100*len(xz)/original_size:.1f}%)")

        # zip
        zp = compress_zip(raw)
        row["zip"] = len(zp)
        print(f"  zip:    {format_size(len(zp)):>10s}  ratio: {len(zp)/original_size:.4f}  ({100*len(zp)/original_size:.1f}%)")

        # Neural
        start = time.time()
        neural = nc.compress(text)
        elapsed = time.time() - start
        row["neural"] = len(neural)
        row["time"] = elapsed
        print(f"  neural: {format_size(len(neural)):>10s}  ratio: {len(neural)/original_size:.4f}  ({100*len(neural)/original_size:.1f}%)  time: {elapsed:.1f}s")

        # Verify roundtrip
        restored = nc.decompress(neural)
        if restored == text:
            print(f"  Roundtrip: OK")
        else:
            print(f"  Roundtrip: FAILED!")

        results.append(row)
        print()

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE (compression ratio = compressed/original)")
    print(f"{'='*70}")
    print(f"{'Sample':<10s} {'Original':>10s} {'gzip':>10s} {'xz':>10s} {'zip':>10s} {'neural':>10s} {'time':>8s}")
    print(f"{'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for r in results:
        orig = r['original']
        print(
            f"{r['name']:<10s} "
            f"{format_size(orig):>10s} "
            f"{r['gzip']/orig:.4f}     "
            f"{r['xz']/orig:.4f}     "
            f"{r['zip']/orig:.4f}     "
            f"{r['neural']/orig:.4f}     "
            f"{r['time']:.1f}s"
        )

    # Markdown table for README
    print(f"\n\nMARKDOWN TABLE FOR README:")
    print(f"| Sample | Original | gzip | xz | zip | Neural (ours) |")
    print(f"|--------|----------|------|----|-----|---------------|")
    for r in results:
        orig = r['original']
        print(
            f"| {r['name']} | {format_size(orig)} | "
            f"{format_size(r['gzip'])} ({r['gzip']/orig:.1%}) | "
            f"{format_size(r['xz'])} ({r['xz']/orig:.1%}) | "
            f"{format_size(r['zip'])} ({r['zip']/orig:.1%}) | "
            f"{format_size(r['neural'])} ({r['neural']/orig:.1%}) |"
        )


if __name__ == "__main__":
    main()
