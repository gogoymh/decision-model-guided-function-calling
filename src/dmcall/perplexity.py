"""DM-Call name for :mod:`dmgfc.perplexity`; both names refer to the same module."""

import sys

from dmgfc import perplexity as _implementation

sys.modules[__name__] = _implementation
