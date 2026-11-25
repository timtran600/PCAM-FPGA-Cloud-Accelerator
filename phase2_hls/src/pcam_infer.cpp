// pcam_infer.cpp — PCam INT8 CNN inference kernel, bring-up version.
//
// A deliberately literal transcription of phase1_software/golden.py. No
// pragmas, no dataflow, no streaming, full feature maps held in on-chip
// arrays. It will synthesize badly and that is fine — the only thing this
// version has to do is produce the same int32 scores as the golden model
// on all 32,768 test tiles in C simulation.
//
// Read before optimizing: every loop here carries a label. Those labels
// are the handles for the pragmas added during design space exploration,
// and each DSE run changes exactly one of them.
//
// Correctness rules this file must not break:
//   1. Rounding is round-half-up: (acc * M0 + (1 << (shift-1))) >> shift.
//      Half-even rounding mismatches golden.py on a handful of tiles out
//      of 32,768, which is painful to find.
//   2. The requantization product uses mul_t (64 bits). A 32-bit
//      intermediate wraps silently.
//   3. Zero padding is a literal zero. Valid only because the activation
//      zero-point is 0.
//   4. ReLU is the lower clamp of the requantization, not a separate
//      stage.
//   5. The FC output is NOT requantized. Two raw int32 accumulators leave
//      the kernel.

#include "pcam_infer.hpp"

// ---------------------------------------------------------------------
// Requantize one accumulator to a uint8 activation.
// ---------------------------------------------------------------------
static px_t requantize(acc_t acc, acc_t m0, acc_t shift) {
    const mul_t prod  = (mul_t)acc * (mul_t)m0;
    const mul_t round = ((mul_t)1) << (shift - 1);
    const mul_t val   = (prod + round) >> shift;

    // Clamp to [0, 255]. The lower bound is the ReLU.
    if (val < (mul_t)ACT_MIN) return (px_t)ACT_MIN;
    if (val > (mul_t)ACT_MAX) return (px_t)ACT_MAX;
    return (px_t)val;
}

// ---------------------------------------------------------------------
// 3x3 stride-1 pad-1 convolution, fused with requantization.
// ---------------------------------------------------------------------
// Templated on dimensions so every array bound is a compile-time
// constant, which synthesis requires. The same body serves all five
// convolutions — one piece of hardware to optimize rather than five.
//
// in   [CIN][H][W]   uint8 activations
// w    [COUT][CIN][3][3] int8
// out  [COUT][H][W]  uint8 activations
template <int CIN, int COUT, int H, int W>
static void conv3x3_requant(const px_t  in[CIN][H][W],
                            const wt_t  w[COUT][CIN][3][3],
                            const acc_t bias[COUT],
                            const acc_t m0[COUT],
                            const acc_t shift[COUT],
                            px_t        out[COUT][H][W]) {
CONV_OC:
    for (int oc = 0; oc < COUT; oc++) {
    CONV_ROW:
        for (int y = 0; y < H; y++) {:
#pragma HLS PIPELINE II=1
            for (int x = 0; x < W; x++) {
                acc_t acc = bias[oc];

            CONV_IC:
                for (int ic = 0; ic < CIN; ic++) {
                CONV_KH:
                    for (int kh = 0; kh < 3; kh++) {
                    CONV_KW:
                        for (int kw = 0; kw < 3; kw++) {
                            const int iy = y + kh - 1;
                            const int ix = x + kw - 1;

                            // Zero pad. Activation zero-point is 0, so
                            // skipping the term IS adding a zero — no
                            // correction needed.
                            if (iy < 0 || iy >= H || ix < 0 || ix >= W)
                                continue;

                            acc += (acc_t)w[oc][ic][kh][kw] *
                                   (acc_t)in[ic][iy][ix];
                        }
                    }
                }

                out[oc][y][x] = requantize(acc, m0[oc], shift[oc]);
            }
        }
    }
}

// ---------------------------------------------------------------------
// 2x2 stride-2 max pool. A comparator tree, no arithmetic.
// ---------------------------------------------------------------------
template <int C, int H, int W>
static void maxpool2x2(const px_t in[C][H][W], px_t out[C][H / 2][W / 2]) {
POOL_C:
    for (int c = 0; c < C; c++) {
    POOL_ROW:
        for (int y = 0; y < H / 2; y++) {
        POOL_COL:
            for (int x = 0; x < W / 2; x++) {
                px_t a = in[c][2 * y][2 * x];
                px_t b = in[c][2 * y][2 * x + 1];
                px_t d = in[c][2 * y + 1][2 * x];
                px_t e = in[c][2 * y + 1][2 * x + 1];
                px_t m = a > b ? a : b;
                px_t n = d > e ? d : e;
                out[c][y][x] = m > n ? m : n;
            }
        }
    }
}

// ---------------------------------------------------------------------
// Global average pool: sum only, then requantize.
// ---------------------------------------------------------------------
// There is no division here. export.py folded the 1/GAP_DIVISOR into
// m0[] for this stage, so hardware sums and shifts. Bound: 36 * 255 =
// 9,180 per channel, nowhere near int32.
template <int C, int H, int W>
static void global_sum_requant(const px_t  in[C][H][W],
                               const acc_t bias[C],
                               const acc_t m0[C],
                               const acc_t shift[C],
                               px_t        out[C]) {
GAP_C:
    for (int c = 0; c < C; c++) {
        acc_t acc = bias[c];
    GAP_ROW:
        for (int y = 0; y < H; y++) {
        GAP_COL:
            for (int x = 0; x < W; x++) {
                acc += (acc_t)in[c][y][x];
            }
        }
        out[c] = requantize(acc, m0[c], shift[c]);
    }
}

// ---------------------------------------------------------------------
// Fully connected. Raw int32 out, no requantization.
// ---------------------------------------------------------------------
template <int IN, int OUT>
static void dense(const px_t  in[IN],
                  const wt_t  w[OUT][IN],
                  const acc_t bias[OUT],
                  score_t     out[OUT]) {
FC_OC:
    for (int oc = 0; oc < OUT; oc++) {
        acc_t acc = bias[oc];
    FC_IC:
        for (int ic = 0; ic < IN; ic++) {
            acc += (acc_t)w[oc][ic] * (acc_t)in[ic];
        }
        out[oc] = (score_t)acc;
    }
}

// =====================================================================
// Top level
// =====================================================================
//
// Weights are copied from the blobs into on-chip arrays once, before any
// tile is processed. The full int8 weight set is 131.2 KB, which fits in
// BRAM/URAM on a U280, so the kernel never re-reads weights from memory
// no matter how many tiles it runs.
//
// Feature maps are declared static to keep them off the stack — csim
// segfaults otherwise, since conv1's output alone is 147 KB.

extern "C" void pcam_infer(const wt_t  *w_blob,
                           const acc_t *p_blob,
                           const px_t  *tiles,
                           score_t     *scores,
                           int          n_tiles) {

    // ---- on-chip weights -------------------------------------------
    static wt_t  c1w[CONV1_OUT][CONV1_IN][3][3];
    static wt_t  c2w[CONV2_OUT][CONV2_IN][3][3];
    static wt_t  c3w[CONV3_OUT][CONV3_IN][3][3];
    static wt_t  c4w[CONV4_OUT][CONV4_IN][3][3];
    static wt_t  c5w[CONV5_OUT][CONV5_IN][3][3];
    static wt_t  fcw[FC_OUT][FC_IN];

    static acc_t c1b[CONV1_OUT], c1m[CONV1_OUT], c1s[CONV1_OUT];
    static acc_t c2b[CONV2_OUT], c2m[CONV2_OUT], c2s[CONV2_OUT];
    static acc_t c3b[CONV3_OUT], c3m[CONV3_OUT], c3s[CONV3_OUT];
    static acc_t c4b[CONV4_OUT], c4m[CONV4_OUT], c4s[CONV4_OUT];
    static acc_t c5b[CONV5_OUT], c5m[CONV5_OUT], c5s[CONV5_OUT];
    static acc_t gpb[GAP_CH],    gpm[GAP_CH],    gps[GAP_CH];
    static acc_t fcb[FC_OUT];

    // ---- feature maps ----------------------------------------------
    static px_t  x0[TILE_C][H1][H1];
    static px_t  a1[CONV1_OUT][H1][H1], p1[CONV1_OUT][H2][H2];
    static px_t  a2[CONV2_OUT][H2][H2], p2[CONV2_OUT][H3][H3];
    static px_t  a3[CONV3_OUT][H3][H3], p3[CONV3_OUT][H4][H4];
    static px_t  a4[CONV4_OUT][H4][H4], p4[CONV4_OUT][H5][H5];
    static px_t  a5[CONV5_OUT][H5][H5];
    static px_t  g[GAP_CH];
    static score_t logits[FC_OUT];

    // -----------------------------------------------------------------
    // Load weights. Index arithmetic must match export.py's layer order
    // exactly: conv1..conv5, gap, fc; within a layer, bias then m0 then
    // shift.
    // -----------------------------------------------------------------
#define LOAD_CONV_W(DST, OFF, COUT, CIN)                                   \
    for (int oc = 0; oc < COUT; oc++)                                      \
        for (int ic = 0; ic < CIN; ic++)                                   \
            for (int kh = 0; kh < 3; kh++)                                 \
                for (int kw = 0; kw < 3; kw++)                             \
                    DST[oc][ic][kh][kw] =                                  \
                        w_blob[(OFF) + ((oc * (CIN) + ic) * 9) + kh * 3 + kw];

#define LOAD_PARAMS(B, M, S, OFF, N)                                       \
    for (int i = 0; i < N; i++) {                                          \
        B[i] = p_blob[PB(OFF, N) + i];                                     \
        M[i] = p_blob[PM(OFF, N) + i];                                     \
        S[i] = p_blob[PS(OFF, N) + i];                                     \
    }

LOAD_WEIGHTS:
    {
        LOAD_CONV_W(c1w, WOFF_CONV1, CONV1_OUT, CONV1_IN)
        LOAD_CONV_W(c2w, WOFF_CONV2, CONV2_OUT, CONV2_IN)
        LOAD_CONV_W(c3w, WOFF_CONV3, CONV3_OUT, CONV3_IN)
        LOAD_CONV_W(c4w, WOFF_CONV4, CONV4_OUT, CONV4_IN)
        LOAD_CONV_W(c5w, WOFF_CONV5, CONV5_OUT, CONV5_IN)

        for (int oc = 0; oc < FC_OUT; oc++)
            for (int ic = 0; ic < FC_IN; ic++)
                fcw[oc][ic] = w_blob[WOFF_FC + oc * FC_IN + ic];

        LOAD_PARAMS(c1b, c1m, c1s, POFF_CONV1, CONV1_OUT)
        LOAD_PARAMS(c2b, c2m, c2s, POFF_CONV2, CONV2_OUT)
        LOAD_PARAMS(c3b, c3m, c3s, POFF_CONV3, CONV3_OUT)
        LOAD_PARAMS(c4b, c4m, c4s, POFF_CONV4, CONV4_OUT)
        LOAD_PARAMS(c5b, c5m, c5s, POFF_CONV5, CONV5_OUT)
        LOAD_PARAMS(gpb, gpm, gps, POFF_GAP,   GAP_CH)

        for (int i = 0; i < FC_OUT; i++) fcb[i] = p_blob[POFF_FC + i];
    }

    // -----------------------------------------------------------------
    // One tile at a time. Pipelining this outer loop is the first thing
    // DSE will look at, but not yet.
    // -----------------------------------------------------------------
TILE_LOOP:
    for (int t = 0; t < n_tiles; t++) {

        // Unpack the tile. CHW, contiguous, 27,648 bytes.
    LOAD_TILE:
        for (int c = 0; c < TILE_C; c++)
            for (int y = 0; y < TILE_H; y++)
                for (int x = 0; x < TILE_W; x++)
                    x0[c][y][x] = tiles[(long)t * TILE_BYTES +
                                        ((c * TILE_H + y) * TILE_W + x)];

        conv3x3_requant<TILE_C,    CONV1_OUT, H1, H1>(x0, c1w, c1b, c1m, c1s, a1);
        maxpool2x2     <CONV1_OUT, H1, H1>            (a1, p1);

        conv3x3_requant<CONV1_OUT, CONV2_OUT, H2, H2>(p1, c2w, c2b, c2m, c2s, a2);
        maxpool2x2     <CONV2_OUT, H2, H2>            (a2, p2);

        conv3x3_requant<CONV2_OUT, CONV3_OUT, H3, H3>(p2, c3w, c3b, c3m, c3s, a3);
        maxpool2x2     <CONV3_OUT, H3, H3>            (a3, p3);

        conv3x3_requant<CONV3_OUT, CONV4_OUT, H4, H4>(p3, c4w, c4b, c4m, c4s, a4);
        maxpool2x2     <CONV4_OUT, H4, H4>            (a4, p4);

        conv3x3_requant<CONV4_OUT, CONV5_OUT, H5, H5>(p4, c5w, c5b, c5m, c5s, a5);

        global_sum_requant<GAP_CH, H5, H5>(a5, gpb, gpm, gps, g);
        dense<FC_IN, FC_OUT>(g, fcw, fcb, logits);

    STORE_SCORES:
        for (int i = 0; i < FC_OUT; i++)
            scores[t * FC_OUT + i] = logits[i];
    }
}