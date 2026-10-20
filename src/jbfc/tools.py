"""Compatibility alias for :mod:`dmfc.tools`; new code should import ``dmfc``."""

import sys

from dmfc import tools as _implementation

sys.modules[__name__] = _implementation
