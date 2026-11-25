// tb_top.cpp — whole-network testbench.
//
// Reads the same files the golden model wrote and compares the kernel's
// int32 scores against test_golden_scores.i32. The pass criterion is
// exact equality on every tile. There is no tolerance and no "close
// enough" — a single differing LSB is a failure, because bit-exactness
// against the golden model is the project's definition of correctness.
//
// Build inputs (paths relative to the repo root, overridable by argv):
//   artifacts/weights/weights.bin
//   artifacts/vectors/test_tiles.bin
//   artifacts/vectors/test_golden_scores.i32
//
// Usage:
//   tb_top [n_tiles] [artifacts_dir]
//
// Start with a small n_tiles while debugging. A mismatch on tile 0 means
// something structural; a mismatch on tile 4,137 only means rounding.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

#include "pcam_infer.hpp"

static std::vector<unsigned char> read_file(const std::string &path) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) {
        printf("ERROR: cannot open %s\n", path.c_str());
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> buf(n);
    if (fread(buf.data(), 1, n, f) != (size_t)n) {
        printf("ERROR: short read on %s\n", path.c_str());
        exit(2);
    }
    fclose(f);
    return buf;
}

// Size of a file in bytes, without reading it. ftell() returns long, which
// is 32-bit on Windows; test_tiles.bin (32,768 x 27,648 B = 906 MB) fits.
static long file_size(const std::string &path) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) {
        printf("ERROR: cannot open %s\n", path.c_str());
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fclose(f);
    return n;
}

// Read only the first n bytes of a file (caller has checked n <= size).
static std::vector<unsigned char> read_prefix(const std::string &path,
                                              size_t n) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) {
        printf("ERROR: cannot open %s\n", path.c_str());
        exit(2);
    }
    std::vector<unsigned char> buf(n);
    if (fread(buf.data(), 1, n, f) != n) {
        printf("ERROR: short read on %s\n", path.c_str());
        exit(2);
    }
    fclose(f);
    return buf;
}

// ---------------------------------------------------------------------
// Split export.py's weights.bin into the two blobs the kernel expects.
// ---------------------------------------------------------------------
// weights.bin is written per layer as [weights int8][bias int32]
// [m0 int32][shift int32], in the order conv1..conv5, gap, fc. GAP has no
// weight array; FC has no m0 or shift. This function must mirror
// export.py::write_bin exactly — if the two disagree, every score is
// wrong and the cause looks like a kernel bug.
static void split_weights(const std::vector<unsigned char> &bin,
                          std::vector<wt_t> &w_blob,
                          std::vector<acc_t> &p_blob) {
    struct Layer { int n_w, n_ch; bool has_w, has_req; };
    const Layer layers[] = {
        {W_CONV1, CONV1_OUT, true,  true },
        {W_CONV2, CONV2_OUT, true,  true },
        {W_CONV3, CONV3_OUT, true,  true },
        {W_CONV4, CONV4_OUT, true,  true },
        {W_CONV5, CONV5_OUT, true,  true },
        {0,       GAP_CH,    false, true },
        {W_FC,    FC_OUT,    true,  false},
    };

    w_blob.clear();
    p_blob.clear();
    size_t off = 0;

    for (const Layer &L : layers) {
        if (L.has_w) {
            for (int i = 0; i < L.n_w; i++)
                w_blob.push_back((wt_t)(signed char)bin[off + i]);
            off += L.n_w;
        }
        // bias always present
        const int n_p = L.has_req ? 3 : 1;      // bias [, m0, shift]
        for (int k = 0; k < n_p; k++) {
            for (int i = 0; i < L.n_ch; i++) {
                int v;
                memcpy(&v, &bin[off + 4 * i], 4);
                p_blob.push_back((acc_t)v);
            }
            off += 4 * L.n_ch;
        }
    }

    printf("weights.bin : %zu B consumed of %zu\n", off, bin.size());
    printf("w_blob      : %zu int8  (expected %d)\n", w_blob.size(), W_TOTAL);
    printf("p_blob      : %zu int32 (expected %d)\n", p_blob.size(), P_TOTAL);

    if (off != bin.size() || (int)w_blob.size() != W_TOTAL ||
        (int)p_blob.size() != P_TOTAL) {
        printf("ERROR: weights.bin layout does not match params.h.\n"
               "       Re-run export.py, or reconcile split_weights()\n"
               "       with export.py::write_bin.\n");
        exit(2);
    }
}

int main(int argc, char **argv) {
    const int n_tiles = (argc > 1) ? atoi(argv[1]) : 16;
    const std::string art = (argc > 2) ? argv[2] : "../../../../../../artifacts";

    // atoi() returns 0 on garbage; 0 tiles would be a vacuous PASS.
    if (n_tiles < 1) {
        printf("ERROR: n_tiles must be >= 1 (argv[1] = '%s')\n",
               argc > 1 ? argv[1] : "");
        return 2;
    }

    printf("=== pcam_infer C simulation ===\n");
    printf("tiles       : %d\n", n_tiles);
    printf("artifacts   : %s\n\n", art.c_str());

    std::vector<wt_t>  w_blob;
    std::vector<acc_t> p_blob;
    split_weights(read_file(art + "/weights/weights.bin"), w_blob, p_blob);

    // avail comes from the file size; only the n_tiles under test are read.
    const std::string tiles_path = art + "/vectors/test_tiles.bin";
    const int avail = (int)(file_size(tiles_path) / TILE_BYTES);
    if (n_tiles > avail) {
        printf("ERROR: only %d tiles in test_tiles.bin\n", avail);
        return 2;
    }
    printf("test_tiles  : %d tiles available\n", avail);

    std::vector<unsigned char> tiles =
        read_prefix(tiles_path, (size_t)n_tiles * TILE_BYTES);
    std::vector<unsigned char> gold_raw =
        read_file(art + "/vectors/test_golden_scores.i32");

    // Exactly one int32 golden score per tile in test_tiles.bin. A length
    // mismatch means the two vector files came from different export runs.
    // Together with n_tiles <= avail above, this also guarantees
    // gold_raw.size() >= 4 * n_tiles for the copy below.
    if (gold_raw.size() != 4 * (size_t)avail) {
        printf("ERROR: test_golden_scores.i32 is %zu B, expected %zu B\n"
               "       (4 B x %d tiles in test_tiles.bin). The vector files\n"
               "       are out of sync; regenerate both from one export.\n",
               gold_raw.size(), 4 * (size_t)avail, avail);
        return 2;
    }
    printf("golden      : %zu scores\n\n", gold_raw.size() / 4);

    // golden scores are the DIFFERENCE logit[1]-logit[0], one int32 each
    std::vector<int> gold(n_tiles);
    for (int i = 0; i < n_tiles; i++)
        memcpy(&gold[i], &gold_raw[4 * i], 4);

    std::vector<px_t> tin(n_tiles * TILE_BYTES);
    for (int i = 0; i < n_tiles * TILE_BYTES; i++) tin[i] = (px_t)tiles[i];

    std::vector<score_t> out(n_tiles * FC_OUT);

    printf("running kernel...\n");
    pcam_infer(w_blob.data(), p_blob.data(), tin.data(), out.data(), n_tiles);

    // -----------------------------------------------------------------
    // Compare. The kernel emits two logits; the golden file holds their
    // difference, which is what the host thresholds.
    // -----------------------------------------------------------------
    int bad = 0, first_bad = -1, compared = 0;
    for (int i = 0; i < n_tiles; i++) {
        compared++;
        const int got = (int)out[2 * i + 1] - (int)out[2 * i];
        if (got != gold[i]) {
            if (first_bad < 0) first_bad = i;
            if (bad < 10)
                printf("  tile %6d: got %12d  golden %12d  diff %d\n",
                       i, got, gold[i], got - gold[i]);
            bad++;
        }
    }

    printf("\n");
    // PASS only if every one of the n_tiles tiles was actually compared.
    if (bad == 0 && compared == n_tiles) {
        printf("PASS: %d/%d tiles bit-exact against the golden model\n",
               n_tiles, n_tiles);
        return 0;
    }

    if (compared != n_tiles) {
        printf("FAIL: compared %d of %d tiles\n", compared, n_tiles);
        return 1;
    }
    printf("FAIL: %d/%d tiles differ (first at %d)\n", bad, n_tiles, first_bad);
    printf("\nWhere to look, in order:\n");
    printf("  all tiles wrong, large diffs -> weights.bin layout; check\n"
           "    split_weights() against export.py::write_bin\n");
    printf("  all tiles wrong, small diffs -> rounding; requantize() must\n"
           "    be round-half-up: (acc*M0 + (1<<(shift-1))) >> shift\n");
    printf("  a few tiles wrong            -> mul_t too narrow, or a\n"
           "    clamp boundary off by one\n");
    printf("  first tile wrong only        -> weight load loop indexing\n");
    printf("\nRun the per-layer testbenches to localize before guessing.\n");
    return 1;
}