"""
Vendored ms-pred common utilities (inference-only subset).

Provides chemistry constants, formula manipulation, fingerprints,
and miscellaneous helpers needed by the ICEBERG DAG model.

All `import ms_pred.common` references in the original code should
resolve through this package.
"""

from .fingerprint import *  # noqa: F401,F403
from .chem_utils import *  # noqa: F401,F403
from .misc_utils import *  # noqa: F401,F403

# Suppress annoying RDKit warnings
from rdkit import RDLogger
_lg = RDLogger.logger()
_lg.setLevel(RDLogger.CRITICAL)
