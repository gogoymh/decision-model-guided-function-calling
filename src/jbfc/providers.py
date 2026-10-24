"""Compatibility alias for :mod:`dmgfc.providers`; new code should import ``dmgfc``."""

import sys

from dmgfc import providers as _implementation

sys.modules[__name__] = _implementation
