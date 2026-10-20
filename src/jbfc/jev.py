"""Compatibility alias for :mod:`dmfc.jev`; new code should import ``dmfc``."""

import sys

from dmfc import jev as _implementation

sys.modules[__name__] = _implementation
