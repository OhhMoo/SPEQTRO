"""
CASCADE 2.0 vendored source code.

Vendored from: https://github.com/patonlab/CASCADE
License: MIT
Original location: CASCADE-2.0/models/Predict_SMILES_FF/modules/

This package contains the inference-only code for CASCADE 2.0, a 3D-GNN
(PAiNN architecture) for predicting 13C NMR chemical shifts from molecular
geometry. All internal imports have been rewritten from ``modules.X`` /
``nfp.X`` to ``speqtro.vendors.cascade.X`` / ``speqtro.vendors.cascade.nfp.X``
so the package can be imported without sys.path hacks.

The pickle file ``preprocessor_orig.p`` shipped with CASCADE was serialized
with the old module paths (``nfp.preprocessing.*``).  To load it correctly,
call :func:`install_pickle_aliases` **before** unpickling.
"""

import sys


def install_pickle_aliases():
    """Register ``sys.modules`` aliases so that ``pickle.load`` can resolve
    the old CASCADE module paths (``nfp.*``, ``modules.*``) to their new
    vendored locations inside ``speqtro.vendors.cascade``.

    This must be called **before** ``pickle.load(preprocessor_orig.p)``.
    """
    from speqtro.vendors.cascade import nfp as _nfp
    from speqtro.vendors.cascade.nfp import preprocessing as _preprocessing
    from speqtro.vendors.cascade.nfp.preprocessing import features as _features
    from speqtro.vendors.cascade.nfp.preprocessing import preprocessor as _preprocessor
    from speqtro.vendors.cascade.nfp.preprocessing import sequence as _sequence
    from speqtro.vendors.cascade.nfp.preprocessing import scaling as _scaling
    from speqtro.vendors.cascade.nfp import layers as _layers
    from speqtro.vendors.cascade.nfp.layers import layers as _layers_layers
    from speqtro.vendors.cascade.nfp.layers import utils as _layers_utils
    from speqtro.vendors.cascade.nfp.layers import wrappers as _layers_wrappers
    from speqtro.vendors.cascade.nfp import models as _models
    from speqtro.vendors.cascade.nfp.models import models as _models_models
    from speqtro.vendors.cascade.nfp.models import losses as _models_losses
    from speqtro.vendors.cascade import pooling as _pooling
    from speqtro.vendors.cascade import bessel_basis as _bessel_basis
    from speqtro.vendors.cascade import segment as _segment

    # Aliases for the old ``nfp.*`` paths (used inside the pickle)
    sys.modules.setdefault("nfp", _nfp)
    sys.modules.setdefault("nfp.preprocessing", _preprocessing)
    sys.modules.setdefault("nfp.preprocessing.features", _features)
    sys.modules.setdefault("nfp.preprocessing.preprocessor", _preprocessor)
    sys.modules.setdefault("nfp.preprocessing.sequence", _sequence)
    sys.modules.setdefault("nfp.preprocessing.scaling", _scaling)
    sys.modules.setdefault("nfp.layers", _layers)
    sys.modules.setdefault("nfp.layers.layers", _layers_layers)
    sys.modules.setdefault("nfp.layers.utils", _layers_utils)
    sys.modules.setdefault("nfp.layers.wrappers", _layers_wrappers)
    sys.modules.setdefault("nfp.models", _models)
    sys.modules.setdefault("nfp.models.models", _models_models)
    sys.modules.setdefault("nfp.models.losses", _models_losses)

    # Aliases for the old ``modules.*`` paths (used when loading the .h5 model)
    sys.modules.setdefault("modules", sys.modules[__name__])
    sys.modules.setdefault("modules.pooling", _pooling)
    sys.modules.setdefault("modules.bessel_basis", _bessel_basis)
    sys.modules.setdefault("modules.segment", _segment)
