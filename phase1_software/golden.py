"""Pure-integer NumPy reference model.

Bit-exact target for the HLS kernel: no floats in the datapath, requant by
integer multiply-and-shift only. Writes artifacts/vectors/ and
artifacts/layer_dumps/.
"""
