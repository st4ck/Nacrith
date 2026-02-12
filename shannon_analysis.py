#!/usr/bin/env python3
"""Calculate Shannon entropy bounds for the 100KB sample and compare to Nacrith."""

import math
import os
from collections import Counter

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "sample_100k.txt")

with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
    text = f.read()

raw = text.encode("utf-8")
n = len(raw)

print(f"File: {SAMPLE_PATH}")
print(f"Size: {n} bytes ({n/1024:.1f} KB)")
print()

# ── 0th-order Shannon entropy (byte-level) ──────────────────────────────
# Treats each byte as an independent symbol. This is the theoretical
# minimum for any compressor that only uses single-byte frequencies.

freq = Counter(raw)
H0 = 0.0
for count in freq.values():
    p = count / n
    H0 -= p * math.log2(p)

min_size_0 = H0 * n / 8  # in bytes (H0 is bits per byte)

print(f"=== 0th-order Shannon Entropy (byte-level) ===")
print(f"Unique byte values: {len(freq)}")
print(f"Entropy H0: {H0:.4f} bits/byte")
print(f"Maximum (uniform): 8.0000 bits/byte")
print(f"Theoretical minimum size: {min_size_0:.0f} bytes ({min_size_0/1024:.1f} KB)")
print()

# ── 1st-order Shannon entropy (bigram / byte-pair) ──────────────────────
# Considers the probability of each byte given the previous byte.
# H1 = Σ P(b_prev) * H(B | b_prev)

bigram_counts = Counter()
for i in range(len(raw) - 1):
    bigram_counts[(raw[i], raw[i + 1])] += 1

prev_counts = Counter()
for i in range(len(raw) - 1):
    prev_counts[raw[i]] += 1

H1 = 0.0
for prev_byte, prev_total in prev_counts.items():
    h_cond = 0.0
    for b in range(256):
        c = bigram_counts.get((prev_byte, b), 0)
        if c > 0:
            p = c / prev_total
            h_cond -= p * math.log2(p)
    H1 += (prev_total / (n - 1)) * h_cond

min_size_1 = H1 * n / 8

print(f"=== 1st-order Shannon Entropy (bigram) ===")
print(f"Entropy H1: {H1:.4f} bits/byte")
print(f"Theoretical minimum size: {min_size_1:.0f} bytes ({min_size_1/1024:.1f} KB)")
print()

# ── 2nd-order Shannon entropy (trigram) ─────────────────────────────────

trigram_counts = Counter()
context2_counts = Counter()
for i in range(len(raw) - 2):
    ctx = (raw[i], raw[i + 1])
    trigram_counts[(ctx, raw[i + 2])] += 1
    context2_counts[ctx] += 1

H2 = 0.0
for ctx, ctx_total in context2_counts.items():
    h_cond = 0.0
    for b in range(256):
        c = trigram_counts.get((ctx, b), 0)
        if c > 0:
            p = c / ctx_total
            h_cond -= p * math.log2(p)
    H2 += (ctx_total / (n - 2)) * h_cond

min_size_2 = H2 * n / 8

print(f"=== 2nd-order Shannon Entropy (trigram) ===")
print(f"Entropy H2: {H2:.4f} bits/byte")
print(f"Theoretical minimum size: {min_size_2:.0f} bytes ({min_size_2/1024:.1f} KB)")
print()

# ── Comparison ──────────────────────────────────────────────────────────

nacrith_size = 15886  # from benchmark
gzip_size = 39965
xz_size = 36332

print(f"{'='*60}")
print(f"COMPARISON")
print(f"{'='*60}")
print(f"{'Method':<30s} {'Size':>10s} {'bits/byte':>10s}")
print(f"{'-'*30} {'-'*10} {'-'*10}")
print(f"{'Original':<30s} {n:>10,} {'8.0000':>10s}")
print(f"{'Shannon 0th-order limit':<30s} {min_size_0:>10,.0f} {H0:>10.4f}")
print(f"{'Shannon 1st-order limit':<30s} {min_size_1:>10,.0f} {H1:>10.4f}")
print(f"{'Shannon 2nd-order limit':<30s} {min_size_2:>10,.0f} {H2:>10.4f}")
print(f"{'gzip -9':<30s} {gzip_size:>10,} {gzip_size*8/n:>10.4f}")
print(f"{'xz -9':<30s} {xz_size:>10,} {xz_size*8/n:>10.4f}")
print(f"{'Nacrith':<30s} {nacrith_size:>10,} {nacrith_size*8/n:>10.4f}")
print()

if nacrith_size < min_size_0:
    pct_below = (1 - nacrith_size / min_size_0) * 100
    print(f"Nacrith is {pct_below:.1f}% BELOW the 0th-order Shannon entropy limit!")
    print(f"This proves it captures higher-order structure (grammar, semantics)")
    print(f"that frequency-based methods cannot exploit.")
else:
    print(f"Nacrith is above the 0th-order Shannon limit.")

if nacrith_size < min_size_1:
    pct_below = (1 - nacrith_size / min_size_1) * 100
    print(f"Nacrith is {pct_below:.1f}% BELOW the 1st-order Shannon entropy limit!")

if nacrith_size < min_size_2:
    pct_below = (1 - nacrith_size / min_size_2) * 100
    print(f"Nacrith is {pct_below:.1f}% BELOW the 2nd-order Shannon entropy limit!")
