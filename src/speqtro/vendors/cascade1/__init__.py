"""
CASCADE 1.0 vendored source code.

Vendored from: https://github.com/patonlab/CASCADE (Jupyternotebook-SMILES branch)
License: MIT
Original location: cascade-Jupyternotebook-SMILES/models/

This package contains the inference-only code for CASCADE 1.0, a message-passing
GNN (old TF1/Keras style) that predicts per-hydrogen 1H NMR chemical shifts from
MMFF94s conformer ensembles with Boltzmann weighting.

All internal imports have been rewritten from ``nfp.X`` to
``speqtro.vendors.cascade1.nfp.X`` so the package can be imported without
sys.path hacks. Keras API calls have been updated for TF2 compatibility.

The pickle file ``preprocessor.p`` shipped with CASCADE 1.0 was serialized with
the old module paths (``nfp.preprocessing.*``). To load it correctly, call
:func:`install_pickle_aliases` **before** unpickling.
"""

import sys


def install_pickle_aliases():
    """Register ``sys.modules`` aliases so that ``pickle.load`` can resolve
    the old CASCADE 1.0 module paths (``nfp.*``) to their new vendored
    locations inside ``speqtro.vendors.cascade1``.

    This must be called **before** ``pickle.load(preprocessor.p)``.
    """
    from speqtro.vendors.cascade1 import nfp as _nfp
    from speqtro.vendors.cascade1.nfp import preprocessing as _preprocessing
    from speqtro.vendors.cascade1.nfp.preprocessing import features as _features
    from speqtro.vendors.cascade1.nfp.preprocessing import preprocessor as _preprocessor
    from speqtro.vendors.cascade1.nfp.preprocessing import sequence as _sequence
    from speqtro.vendors.cascade1.nfp.preprocessing import scaling as _scaling
    from speqtro.vendors.cascade1.nfp import layers as _layers
    from speqtro.vendors.cascade1.nfp.layers import layers as _layers_layers
    from speqtro.vendors.cascade1.nfp.layers import utils as _layers_utils
    from speqtro.vendors.cascade1.nfp.layers import wrappers as _layers_wrappers
    from speqtro.vendors.cascade1.nfp import models as _models
    from speqtro.vendors.cascade1.nfp.models import models as _models_models
    from speqtro.vendors.cascade1.nfp.models import losses as _models_losses

    # Aliases for the old ``nfp.*`` paths (used inside the pickle)
    # Use setdefault so we don't clobber CASCADE 2.0 aliases if both are loaded.
    # Since CASCADE 1.0 and 2.0 share the same top-level ``nfp`` namespace in
    # the original source, we register cascade1 under a distinct key to avoid
    # conflicts.  The preprocessor.p for CASCADE 1.0 was also pickled with
    # ``nfp.*`` paths, so we must register these aliases.  If CASCADE 2.0 has
    # already registered them, we overwrite with cascade1's modules here since
    # this function is only called when loading CASCADE 1.0 models.
    sys.modules["nfp"] = _nfp
    sys.modules["nfp.preprocessing"] = _preprocessing
    sys.modules["nfp.preprocessing.features"] = _features
    sys.modules["nfp.preprocessing.preprocessor"] = _preprocessor
    sys.modules["nfp.preprocessing.sequence"] = _sequence
    sys.modules["nfp.preprocessing.scaling"] = _scaling
    sys.modules["nfp.layers"] = _layers
    sys.modules["nfp.layers.layers"] = _layers_layers
    sys.modules["nfp.layers.utils"] = _layers_utils
    sys.modules["nfp.layers.wrappers"] = _layers_wrappers
    sys.modules["nfp.models"] = _models
    sys.modules["nfp.models.models"] = _models_models
    sys.modules["nfp.models.losses"] = _models_losses
