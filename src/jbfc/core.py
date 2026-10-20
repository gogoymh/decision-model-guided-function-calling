"""Compatibility alias for :mod:`dmfc.core`; new code should import ``dmfc``."""

import sys

from dmfc import core as _implementation

sys.modules[__name__] = _implementation
