"""NFC data tables for the Kev encoder (common/decision-nfc-data.cpp).

Uses the unicodedata module of the Python that runs it and writes:
  - full canonical decompositions (recursively expanded, singletons included)
  - canonical combining class ranges
  - canonical composition pairs (verified against unicodedata NFC, which drops
    the composition exclusions)

Hangul is not stored: it is decomposed and composed algorithmically in C++.

    python3 gen_nfc_data.py [OUTPUT_CPP]
"""
import os
import sys
import unicodedata

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, "common", "decision-nfc-data.cpp")

HANGUL_SYL_START = 0xAC00
HANGUL_SYL_END = 0xD7A3

MAX_CODEPOINT = 0x110000


def canonical_parts(ch):
    d = unicodedata.decomposition(ch)
    if not d or d.startswith("<"):
        return None
    return [int(p, 16) for p in d.split()]


decomp_rows = []
for cp in range(MAX_CODEPOINT):
    parts = canonical_parts(chr(cp))
    if parts is None or HANGUL_SYL_START <= cp <= HANGUL_SYL_END:
        continue
    full = [ord(c) for c in unicodedata.normalize("NFD", chr(cp))]
    assert HANGUL_SYL_START > full[0] or full[0] > HANGUL_SYL_END  # no Hangul inside decompositions
    decomp_rows.append((cp, full))
decomp_rows.sort(key=lambda r: r[0])

comp_rows = []
for cp in range(MAX_CODEPOINT):
    parts = canonical_parts(chr(cp))
    if parts is None or len(parts) != 2 or HANGUL_SYL_START <= cp <= HANGUL_SYL_END:
        continue
    a, b = parts
    if unicodedata.normalize("NFC", chr(a) + chr(b)) == chr(cp):
        assert unicodedata.combining(chr(cp)) == 0  # composites are starters
        comp_rows.append((a, b, cp))
comp_rows.sort()

ccc_rows = []
prev = None
for cp in range(MAX_CODEPOINT):
    c = unicodedata.combining(chr(cp))
    if c == 0:
        prev = None
        continue
    if prev is not None and prev[2] == c and prev[1] + 1 == cp:
        prev[1] = cp
    else:
        prev = [cp, cp, c]
        ccc_rows.append(prev)

data_pool = []
decomp_entries = []
for cp, full in decomp_rows:
    decomp_entries.append((cp, len(data_pool), len(full)))
    data_pool.extend(full)

with open(OUT, "w") as f:
    f.write("// generated with tools/kev/golden/gen_nfc_data.py (unicodedata %s) - do not edit\n\n" % unicodedata.unidata_version)
    f.write('#include "decision-nfc-data.h"\n\n')
    f.write("namespace kev_nfc {\n\n")
    f.write("// sorted by cpt: cpt, offset into decomp_data, length\n")
    f.write("const decomp_entry decomp_table[] = {\n")
    for cp, off, ln in decomp_entries:
        f.write("{0x%06X, 0x%04X, %d},\n" % (cp, off, ln))
    f.write("};\n")
    f.write("const size_t decomp_size = sizeof(decomp_table) / sizeof(decomp_table[0]);\n\n")
    f.write("const uint32_t decomp_data[] = {\n")
    for i in range(0, len(data_pool), 8):
        f.write("".join("0x%06X, " % c for c in data_pool[i:i + 8]) + "\n")
    f.write("};\n")
    f.write("const size_t decomp_data_size = sizeof(decomp_data) / sizeof(decomp_data[0]);\n\n")
    f.write("// sorted by first, then second: first, second, composite\n")
    f.write("const comp_entry comp_table[] = {\n")
    for a, b, cp in comp_rows:
        f.write("{0x%06X, 0x%06X, 0x%06X},\n" % (a, b, cp))
    f.write("};\n")
    f.write("const size_t comp_size = sizeof(comp_table) / sizeof(comp_table[0]);\n\n")
    f.write("// sorted by first: first, last, ccc (0 outside the ranges)\n")
    f.write("const ccc_entry ccc_table[] = {\n")
    for lo, hi, c in ccc_rows:
        f.write("{0x%06X, 0x%06X, %d},\n" % (lo, hi, c))
    f.write("};\n")
    f.write("const size_t ccc_size = sizeof(ccc_table) / sizeof(ccc_table[0]);\n\n")
    f.write("}\n")

print("wrote %s: %d decompositions, %d composition pairs, %d ccc ranges (unicodedata %s)" % (
    OUT, len(decomp_entries), len(comp_rows), len(ccc_rows), unicodedata.unidata_version))
