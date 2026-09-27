#pragma once

// Unicode data for the Kev NFC normalizer, generated with
// tools/kev/golden/gen_nfc_data.py (see the file header for the Unicode version).
// Hangul decomposition and composition are algorithmic, not stored here.

#include <cstddef>
#include <cstdint>

namespace kev_nfc {

struct decomp_entry {
    uint32_t cpt;     // codepoint with a canonical decomposition
    uint32_t offset;  // start of the expansion in decomp_data
    uint32_t len;     // length of the expansion
};

struct comp_entry {
    uint32_t first;
    uint32_t second;
    uint32_t composite;
};

struct ccc_entry {
    uint32_t first;
    uint32_t last;
    uint32_t ccc;
};

extern const decomp_entry decomp_table[];
extern const size_t      decomp_size;
extern const uint32_t    decomp_data[];
extern const size_t      decomp_data_size;
extern const comp_entry  comp_table[];
extern const size_t      comp_size;
extern const ccc_entry   ccc_table[];
extern const size_t      ccc_size;

}
