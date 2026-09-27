"""Write a vocab-only GGUF (all metadata KV, zero tensors) from a full Kev GGUF, like llama.cpp's models/ggml-vocab-*.gguf.
Usage: PYTHONPATH=<llama.cpp>/gguf-py python vocab_only.py IN.gguf OUT.gguf"""
import sys
import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType

src, dst = sys.argv[1], sys.argv[2]
r = GGUFReader(src)
arch = r.fields["general.architecture"].contents()
w = GGUFWriter(dst, arch)
skip = {"general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
n = 0
for name, f in r.fields.items():
    if name in skip: continue
    vt = f.types[0]
    if vt == GGUFValueType.ARRAY:
        w.add_key_value(name, f.contents(), vt, sub_type=f.types[-1])
    else:
        w.add_key_value(name, f.contents(), vt)
    n += 1
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"{dst}: {n} keys, 0 tensors")
