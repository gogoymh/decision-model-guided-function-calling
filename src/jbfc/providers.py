"""Compatibility alias for :mod:`dmfc.providers`; new code should import ``dmfc``."""

import sys

from dmfc import providers as _implementation

sys.modules[__name__] = _implementation
