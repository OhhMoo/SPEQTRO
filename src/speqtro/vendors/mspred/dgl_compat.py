"""
DGL 2.x compatibility shims for ms-pred vendored code.

ms-pred was written against DGL 0.8. This module provides compatibility
shims so that the vendored code works with DGL 2.x.

Changes from DGL 0.8 -> 2.x:
  - `dgl.nn.expand_as_pair` may be removed; we inline it here.
  - `dgl.backend.pytorch.pad_packed_tensor` and `pack_padded_tensor`
    are internal DGL helpers that may not exist; we provide pure-torch
    replacements in nn_utils.py instead.
  - `dgl.EID` constant may be removed; we fall back to the string '_ID'.
"""

import dgl

# -- expand_as_pair -----------------------------------------------------------
# In DGL 0.8 this lived in dgl.nn; in DGL 2.x it may be absent.
try:
    from dgl.nn import expand_as_pair  # noqa: F401
except ImportError:
    # DGL 2.x compat: inline the trivial implementation.
    def expand_as_pair(input, g=None):  # noqa: A002
        """Return (input, input) -- the homogeneous-graph case."""
        return (input, input)


# -- dgl.EID ------------------------------------------------------------------
# In DGL 0.8 `dgl.EID` was a constant string '_ID'.
# In DGL 2.x it may still exist, but we provide a safe fallback.
try:
    EID = dgl.EID
except AttributeError:
    # DGL 2.x compat: use the raw string that DGL stores internally.
    EID = "_ID"
