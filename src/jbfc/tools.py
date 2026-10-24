"""Compatibility alias for :mod:`dmgfc.tools`; new code should import ``dmgfc``."""

import sys

from dmgfc import tools as _implementation

sys.modules[__name__] = _implementation
