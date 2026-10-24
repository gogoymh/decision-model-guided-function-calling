"""Compatibility alias for :mod:`dmgfc.core`; new code should import ``dmgfc``."""

import sys

from dmgfc import core as _implementation

sys.modules[__name__] = _implementation
