"""Compatibility alias for :mod:`dmgfc._prompt`; new code should import ``dmgfc``."""

import sys

from dmgfc import _prompt as _implementation

sys.modules[__name__] = _implementation
