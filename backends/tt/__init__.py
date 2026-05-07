"""Tenstorrent (tt-metal) backend.

Layout:
    backend.py            — Python wrapper around the daemon subprocess
    tt-metal/             — vendored tt-metal SDK (gitignored except our kernel)
        tt_metal/programming_examples/gaussian_splatting/   ← our C++ kernels

The wrapper talks to the daemon via stdin/stdout + .npy files; see
backend.py for the protocol.
"""
