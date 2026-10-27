"""Dump INT8 weights, scales and zero-points to artifacts/weights/.

Also emits phase2_hls/src/params.h, so the HLS kernel and the golden model
are generated from one source of truth.
"""
