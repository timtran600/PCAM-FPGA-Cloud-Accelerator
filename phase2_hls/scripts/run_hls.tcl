# run_hls.tcl — Vitis HLS project for the PCam inference kernel.
#
# DEPRECATED (2026-11-25). New work uses the unified Vitis component flow
# in phase2_hls/workspace/ (see phase2_hls/README.md). The baseline
# component workspace/pcam_base reproduced this script's sol1 csim
# (16/16 bit-exact) and its synthesis report line-for-line. This script and
# pcam_infer_prj/ are kept only as the reference that migration was
# verified against. Do not use them for new DSE runs.
#
#   vitis_hls -f run_hls.tcl                  csim only (the default)
#   vitis_hls -f run_hls.tcl -tclargs synth   csim + synthesis
#   vitis_hls -f run_hls.tcl -tclargs cosim   csim + synthesis + cosim
#   vitis_hls -f run_hls.tcl -tclargs csim 1024   csim on 1024 tiles
#   vitis_hls -f run_hls.tcl -tclargs csim 16 O   csim with an optimized
#                                                 (-O2 gcc) build
#   vitis_hls -f run_hls.tcl -tclargs synth sol=sol2
#                                     csim + synthesis into solution sol2
#
# Args: <stage> [tiles] [O] [sol=NAME]. Any token starting with "sol="
# selects the solution (default sol1); it may appear anywhere after the
# stage (see the parsing comment for the Windows '=' split). NAME must
# match ^[A-Za-z0-9_]+$. The token is removed before the positional args
# below are parsed. The project is opened without -reset,
# so other solutions survive; only the named solution is reset.
#
# Positional args: <stage> <tiles> [O]. stage is one of csim|synth|cosim. The third
# arg "O" is opt-in, allowed only with stage csim, and only changes the
# C-simulation compiler flags (csim_design -O plus -O2 in the gcc cflags;
# see the add_files comment for why -O alone is not enough). It is not a
# pragma. Default stays the debug build.
#
# Run from phase2_hls/scripts/. Paths below are relative to that.
#
# Development happens on 2023.2 locally; every number that goes into
# DSE_LOG.md or the report must come from 2023.2 on the OCT build
# machine, because latency, resource and Fmax results move between tool
# versions and the .xclbin has to match the card's platform shell.

# vitis_hls 2023.2 puts "-f run_hls.tcl" at the front of $argv, ahead of
# anything after -tclargs. Strip it so args[0] is the stage, args[1] tiles.
set args $argv
if {[lindex $args 0] eq "-f"} { set args [lrange $args 2 end] }

# Pull out the solution selector; everything else stays positional.
# On Windows, vitis_hls.bat re-splits its args with batch %1 semantics,
# which treat '=' as a separator, so "sol=sol2" arrives here as the two
# tokens "sol" "sol2". Accept both that split form and the literal
# "sol=NAME" (Linux launcher).
set SOL "sol1"
set pos {}
for {set i 0} {$i < [llength $args]} {incr i} {
    set a [lindex $args $i]
    if {[string match "sol=*" $a]} {
        set SOL [string range $a 4 end]
    } elseif {$a eq "sol"} {
        incr i
        if {$i >= [llength $args]} { error "'sol' needs a solution name (sol=NAME)" }
        set SOL [lindex $args $i]
    } else {
        lappend pos $a
        continue
    }
    if {![regexp {^[A-Za-z0-9_]+$} $SOL]} {
        error "solution name must match ^\[A-Za-z0-9_\]+\$, got '$SOL'"
    }
}
set args $pos

set stage "csim"
if {[llength $args] > 0} { set stage [lindex $args 0] }
if {$stage ni {csim synth cosim}} {
    error "stage must be one of csim, synth, cosim, got '$stage'"
}

set CSIM_OPT 0                         ;# 1 -> csim_design -O
if {[llength $args] > 2} {
    if {[lindex $args 2] ne "O"} {
        error "third arg must be 'O' (optimized csim) or omitted, got '[lindex $args 2]'"
    }
    # The O flag adds -O2 to the add_files cflags, and those cflags
    # also feed the synthesis front end, so allow O only for csim.
    if {$stage ne "csim"} {
        error "'O' (optimized csim) is only allowed with stage csim, got '$stage'"
    }
    set CSIM_OPT 1
}
if {[llength $args] > 3} {
    error "too many args: '$args' (expected <stage> <tiles> \[O\] \[sol=NAME\])"
}

# U280: xcu280-fsvh2892-2L-e. The part alone is enough for synthesis —
# no platform needed, which is why this phase runs anywhere.
set PART   "xcu280-fsvh2892-2L-e"
set PERIOD 3.33                        ;# 300 MHz target

set TILES_CSIM 16                      ;# keep small while debugging
if {[llength $args] > 1} { set TILES_CSIM [lindex $args 1] }
if {![string is integer -strict $TILES_CSIM] || $TILES_CSIM < 1} {
    error "tile count must be a positive integer, got '$TILES_CSIM'"
}

# Absolute paths: csim/cosim run the tb from <sol>/csim/build (or
# <sol>/sim/wrapc), so anything relative to scripts/ would not resolve.
set ART [file normalize ../../artifacts]
set SRC [file normalize ../src]

# No -reset here: it would delete every solution in the project. Only
# the selected solution is reset, by open_solution below.
open_project pcam_infer_prj
set_top pcam_infer

# In 2023.2 (Windows, gcc), csim_design -O only switches the Makefile to
# "release mode". The generated csim.mk compile rule never uses the -O1
# that mode sets, so the binary is still built at gcc's default -O0. The
# per-file -cflags string does reach gcc verbatim, so the opt-in build
# puts -O2 there. This is a compiler flag, not a pragma or directive.
set CFLAGS "-I$SRC -I$ART/weights -std=c++14"
if {$CSIM_OPT} { append CFLAGS " -O2" }

add_files ../src/pcam_infer.cpp -cflags $CFLAGS
add_files -tb ../tb/tb_top.cpp -cflags $CFLAGS

puts "=== SOLUTION: $SOL ==="
open_solution -reset $SOL -flow_target vitis
set_part $PART
create_clock -period $PERIOD -name default

# No directives in any solution; pragmas live in the source. Each DSE
# run gets its own solution name (sol=NAME) and changes exactly one
# pragma, logged in ../reports/DSE_LOG.md.

if {$CSIM_OPT} {
    puts "=== C SIMULATION (optimized build, -O) ==="
    csim_design -O -argv "$TILES_CSIM $ART"
} else {
    puts "=== C SIMULATION (debug build) ==="
    csim_design -argv "$TILES_CSIM $ART"
}

if {$stage eq "synth" || $stage eq "cosim"} {
    puts "=== SYNTHESIS ==="
    csynth_design
}

if {$stage eq "cosim"} {
    # Cosim is far slower than csim. Two tiles is enough to confirm the
    # RTL matches the C; full-split verification stays in csim.
    puts "=== COSIMULATION ==="
    cosim_design -argv "2 $ART" -trace_level none
}

exit