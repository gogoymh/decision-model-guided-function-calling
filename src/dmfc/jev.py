"""Compatibility alias for :mod:`dmgfc.jev`; new code should import ``dmgfc``."""

import sys

from dmgfc import jev as _implementation

sys.modules[__name__] = _implementation
