"""Compatibility alias for :mod:`dmfc._prompt`; new code should import ``dmfc``."""

import sys

from dmfc import _prompt as _implementation

sys.modules[__name__] = _implementation
