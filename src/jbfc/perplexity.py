"""Compatibility alias for :mod:`dmgfc.perplexity`; new code should import ``dmgfc``."""

import sys

from dmgfc import perplexity as _implementation

sys.modules[__name__] = _implementation
