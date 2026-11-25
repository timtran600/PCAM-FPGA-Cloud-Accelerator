# phase2_hls — HLS kernel, unified Vitis 2023.2 component flow

The kernel source is `src/pcam_infer.{cpp,hpp}`, which has no pragmas. The
whole-network C simulation testbench is `tb/tb_top.cpp`, and it compares
every tile bit-exactly against `artifacts/vectors/test_golden_scores.i32`.

Each design point (the baseline and every DSE variant) is its own **HLS
component** in the unified Vitis workspace `workspace/`. A component is
configured by an `hls_config.cfg` file, not by a Tcl script.

```
phase2_hls/
  src/                      shared kernel source (all components compile it)
  tb/                       shared testbench
  workspace/                unified Vitis workspace  (vitis -w workspace)
    pcam_base/              baseline component, run 1, no directives
      hls_config.cfg        the component's configuration (tracked)
      vitis-comp.json       component descriptor (tracked)
      pcam_base/            build output (work_dir, gitignored)
    pcam_dseNN/             one directory per DSE variant
  reports/DSE_LOG.md        one row per synthesis run
  scripts/run_hls.tcl       DEPRECATED classic Tcl flow (kept for reference)
```

## Environment

Use Vitis **2023.2** only (CLAUDE.md §3). Open a shell where it is set up:

```
call C:\Xilinx\Vitis\2023.2\settings64.bat
```

Do this in a plain shell, not one with the Python `pcam_env` venv activated.

Open the workspace in the IDE: `vitis -w C:\GitRepos\PCAMCNN\phase2_hls\workspace`.
Every command below also works headless and edits the same files the IDE uses.

## The baseline component (`workspace/pcam_base/hls_config.cfg`)

| Key | Value | Why |
|---|---|---|
| `part` | `xcu280-fsvh2892-2L-e` | U280 |
| `clock` | `3.33ns` | 300 MHz target |
| `flow_target` | `vitis` | **Must be set.** The unified default is `vivado`, which changes interface defaults and the numbers. |
| `syn.top` | `pcam_infer` | |
| `syn.file` / `tb.file` | `../../src/pcam_infer.cpp`, `../../tb/tb_top.cpp` | Relative to the config file's own directory, so they point at the shared source. |
| `syn.file_cflags` | `-I…/phase2_hls/src -I…/artifacts/weights -std=c++14` | Absolute `-I` paths, because compiles run from deep build directories. |
| `syn.file_csimflags` | `-O2` | Applies to C simulation only. It does not reach synthesis. |
| `tb.file_cflags` | same as `syn.file_cflags`, plus `-O2` | The testbench is only ever compiled for csim. |
| `csim.O` | `true` | Optimized csim build ("release mode"). |
| `csim.argv` | `16 C:/GitRepos/PCAMCNN/artifacts` | Tile count, then the absolute artifacts path. |
| `package.output.format` / `package.output.syn` | `xo` / `true` | Tool defaults. `v++ -c` packages an `.xo` after synthesis. |

`-O2` comes from the csim flags. `csim.O` alone is not enough: in 2023.2 the
generated csim makefile drops the optimisation level, so without `-O2` the
build runs at gcc's `-O0`.

The baseline has **no `syn.directive.*` lines**. It must stay that way.

## Run csim (bit-exact check against the golden model)

From the component directory:

```
cd /d C:\GitRepos\PCAMCNN\phase2_hls\workspace\pcam_base
vitis-run --mode hls --csim --config hls_config.cfg --work_dir pcam_base
```

- **Pass:** `PASS: N/N tiles bit-exact against the golden model` and
  `CSim done with 0 errors`.
- **Fail:** any `FAIL` line. Stop and report the first divergent tile. Never relax the
  comparison.
- **Tile count:** edit the first token of `csim.argv`. 16 is for quick checks.
  1,024 takes about 14 min at -O2.

`--work_dir` must equal the `work_dir` in that component's `vitis-comp.json`,
which is the component name. That way the IDE and the command line share
one build directory.

## Synthesize

```
cd /d C:\GitRepos\PCAMCNN\phase2_hls\workspace\pcam_base
v++ -c --mode hls --config hls_config.cfg --work_dir pcam_base
```

The report lands in `<component>/<work_dir>/hls/syn/report/pcam_infer_csynth.rpt`.
Copy the numbers from there into `reports/DSE_LOG.md`.

## Creating DSE variant N

Each variant is a copy of its parent component that differs **only** by
directives in its own `hls_config.cfg`. CLAUDE.md allows one change per
synthesis run. Name variants `pcam_dseNN` to match the DSE_LOG run number.

### 1. Clone the parent

After cloning, make sure the clone has no copy of the parent's build directory
(`<parent>/<parent>/`), and delete one if it is there. Clone and config
behaviour below was verified with the 2023.2 Python CLI. The IDE's own clone
menu entry was not tested.

- **Headless (verified):** run `vitis -s clone.py` with:

  ```python
  import vitis
  c = vitis.create_client()
  c.set_workspace(r"C:/GitRepos/PCAMCNN/phase2_hls/workspace")
  c.clone_component(name="pcam_base", new_name="pcam_dse03")
  vitis.dispose()
  ```

  The clone receives its own copy of `hls_config.cfg`, and a
  `vitis-comp.json` with its own `name` and `work_dir`. `vitis.dispose()`
  can leave a `RigelApp` Java server running that holds `workspace/.lock`.
  If a later `vitis` run reports the workspace "already in use", end that
  process.

- **By hand:** this is equivalent. Copy `hls_config.cfg` and `vitis-comp.json` into
  `workspace/pcam_dseNN/`, and set `"name"` and `"work_dir"` in the JSON to
  `pcam_dseNN`. Do not copy the parent's build directory.

### 2. Add exactly one change

Add a `syn.directive.*` line to the variant's `hls_config.cfg`. The value
syntax is the one the installed IDE schema documents. For example:

```
syn.directive.pipeline=location=conv3x3_requant/CONV_COL II=1
```

Locations use the loop labels in `src/pcam_infer.cpp` (`CONV_OC`, `CONV_ROW`,
`CONV_COL`, `CONV_IC`, `CONV_KH`, `CONV_KW`, `TILE_LOOP`, `LOAD_TILE`, …).
The directive lives only in that variant's config. Every other component is
unaffected.

Check the first synthesis log of each new variant for its `Applying ini
'syn.directive…'` line. Then check the effect it should have, such as the
loop's pipelining result. As of the migration, the directive syntax was
confirmed to be accepted by the config parser, but its effect has not yet
been confirmed in a synthesis run.

### 3. Run csim, then synthesize

Use the two commands above with `--work_dir pcam_dseNN`. A directive must
never change results. Re-verify with csim (1,024 tiles) and record
bit-exact yes or no in the log.

### 4. Log the row

Add the row to `reports/DSE_LOG.md`:
- the component name;
- its parent;
- the directive line, verbatim;
- the source revision it was built from.

## What is shared, and what that implies

All components read the **same** `src/` and `tb/` at build time. Nothing is
copied. As a result:

- **A directive change is private to one variant.** That is why every DSE knob
  goes in `syn.directive.*`, never as a `#pragma` in the shared source.
- **A source edit changes every component the next time it is built**, past
  variants included. Re-synthesizing an old variant after the source has
  changed does **not** reproduce its DSE_LOG row.
- **Structural source changes** (line buffers, DATAFLOW restructuring) need care:
  commit the source first, and record the commit hash in each DSE_LOG row.
  To reproduce an old row, check out that commit and rebuild the component.

## Classic flow (deprecated)

`scripts/run_hls.tcl` and its `scripts/pcam_infer_prj/` (`sol1` = run 1
baseline) still work, and are kept only as the reference the migration was
verified against. Use the components above for new work.
