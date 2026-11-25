# DSE Log — pcam_infer

One pragma change per synthesis run (CLAUDE.md §1). Every number below comes
from Vitis HLS 2023.2 (Build 4023990) synthesis reports, part
`xcu280-fsvh2892-2L-e`, target clock 3.33 ns (300 MHz), flow target `vitis`.

Reproduce a row: from `phase2_hls/scripts/`,
`vitis_hls -f run_hls.tcl -tclargs synth` (runs a 16-tile debug csim, then
`csynth_design`). Source of truth per run:
`pcam_infer_prj/sol1/syn/report/pcam_infer_csynth.rpt`.
Run 1 (baseline) reports are archived in `run1_baseline_pcam_infer_csynth.rpt` and `run1_baseline_csynth.rpt` (pcam_infer_prj/ is gitignored).

Column notes:
- **Latency** is the per-tile `TILE_LOOP` iteration latency in cycles. The
  top-level latency reports `?` because the tile count is a runtime argument.
- **II** is the top-level interval. `TILE_LOOP` is not pipelined, so the
  interval between tiles equals the per-tile latency.
- Utilization percentages are of the whole U280 device. The per-SLR figure is in
  the notes.
- **Fmax** is the HLS estimate (`HLS 200-789`), not a post-route result.

| Run | What changed | Latency (cycles/tile) | II | LUT | FF | BRAM_18K | URAM | DSP | Fmax (est.) | Bit-exact | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Baseline, no pragmas | 31,549,338 | not pipelined (= latency) | 63,254 (4%) | 276,350 (10%) | 292 (7%) | 0 (0%) | 69 (~0%) | 356.13 MHz (2.808 ns est. vs 3.33 ns target) | yes (1,024 tiles, csim -O2) | See run 1 notes below. |
| 2 (**reverted**) | `#pragma HLS PIPELINE` on TILE_LOOP | — | — | — | — | — | — | — | — | yes (1,024 tiles, csim -O2) | Front end stalled past 45 min at 9,568 instructions after Compile/Link; killed with no schedule produced. Whole-network pipelining is infeasible while the five convolutions share static feature-map arrays, since tile n+1 cannot overlap tile n. Motivates the structural change to streamed line buffers plus DATAFLOW. Pragma reverted; source back to zero pragmas. See run 2 notes below. |

## Run 1 notes (2026-11-24, baseline)

- **Full-split status:** full 32,768-tile verification is pending a working
  fast native build. The native `-O2` g++ 8.3.0 build ran at 0.79 s/tile.
  It passed 256/256 bit-exact, but the full run was stopped unfinished.
- **Per-SLR utilization:** BRAM 21%, DSP 2%, FF 31%, LUT 14%, URAM 0%.
- **Timing warning:**
  `HLS 200-871`: estimated clock period 2.808 ns exceeds the effective
  delay budget of 2.431 ns (3.33 ns target minus 0.899 ns uncertainty).
- **No pragmas, but tool defaults still apply.** The source has no pragmas
  or directives, but the `vitis` flow's default auto-pipelining still
  pipelined inner loops:
  - conv1 `CONV_OC_CONV_ROW_CONV_COL` reached II=1 with 59 DSP.
  - Several load and copy loops reached II=1.
  - `VITIS_LOOP_239_24` and `VITIS_LOOP_240_25` reached II=3, limited by
    `gmem` ports (`HLS 200-885`).
- **Dominant stages** (cycles per tile):

  | Stage | Cycles |
  |---|---|
  | conv2 (16→32 @48×48) | 11,870,211 |
  | conv3 (32→64 @24×24) | 11,243,523 |
  | conv4 (64→64 @12×12) | 5,465,091 |
  | conv5 (64→128 @6×6) | 2,727,939 |
  | conv1 (3→16 @96×96) | 147,506 |

- **Warnings:** 0 critical warnings. The other 70 warnings by ID:

  | ID | Count | Kind |
  |---|---|---|
  | HLS 200-1016 | 6 | critical-path report |
  | HLS 200-871 | 6 | clock estimate over budget |
  | HLS 200-885 | 10 | II violation from gmem ports |
  | HLS 200-960 | 5 | cannot flatten `CONV_COL` or `FC_OC` (imperfect loop nest) |
  | XFORM 203-561 | 27 | loop bound updates |
  | SYN 201-103 | 11 | name legalization |
  | RTGEN 206-101 | 5 | AXI reset and power-on register init |

## Run 2 notes (2026-11-24, PIPELINE on TILE_LOOP)

- **Change:** one line, `#pragma HLS PIPELINE`, as the first statement in
  the `TILE_LOOP` body (pcam_infer.cpp:254). No other pragmas or
  directives. Solution `sol2`, run with `-tclargs synth sol=sol2`.
- **Result: synthesis did not complete.** The 16-tile debug csim passed:
  `PASS: 16/16 tiles bit-exact against the golden model`, with csim
  elapsed time 69.572 s. Then `csynth_design` started at about 20:47 and
  passed the Compile/Link phase:
  `INFO: [HLS 200-1995] There were 9,568 instructions in the design after the 'Compile/Link' phase of compilation.`
  It printed nothing more until a watchdog killed `vitis_hls` and its
  child processes at 21:32:50, after 45 min in csynth. The last log line
  is the HLS 200-1995 line above. The run printed no error, critical
  warning, `Unable to enforce`, II-violation, unroll, or partition
  message before the kill.
- **Where it stalled:** the partial `sol2/syn/report/csynth_design_size.rpt`
  says (the run 2 csim then reset `sol2`, so this file was deleted; it
  was only kept in a temporary backup) `* C-Synthesis has not completed!` and lists the `Unroll/Inline`
  phase as `pending`. Every phase after that is blank. PIPELINE on an
  outer loop asks the tool to fully unroll every loop nested inside it.
  Here that covers all five convolutions, about 33.2 MMAC per tile. The
  tool never got past that unroll.
- **Memory:** tree private memory peaked at 1.24 GB and grew slowly, by
  about 0.1 GB every 7 min. It was nowhere near the 32 GB kill limit, so
  the time limit stopped the run, not memory.
- **No II, latency, utilization or Fmax numbers exist for this run.** The
  tool never produced `pcam_infer_csynth.rpt`.
- **Finding:** with a coarse-grained PIPELINE on `TILE_LOOP`, 2023.2 did
  not get past Unroll/Inline in 45 min. The loop body works on whole feature maps in
  shared on-chip arrays, so pipelining at tile level has nothing to
  overlap. All it does is force a full unroll of the per-tile compute.
- **Bit-exact:** csim on the source with the pragma, run with
  `-tclargs csim 1024 O sol=sol2`, printed
  `PASS: 1024/1024 tiles bit-exact against the golden model` and
  `CSim done with 0 errors` (988.073 s).
- **Reverted (2026-11-24):** the pragma was removed from `TILE_LOOP`, and
  `phase2_hls/src/` again has zero pragmas. The run 1 baseline
  (`sol1`) remains the reference for the next run.
