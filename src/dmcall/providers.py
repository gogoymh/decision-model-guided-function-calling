"""DM-Call name for :mod:`dmgfc.providers`; both names refer to the same module."""

import sys

from dmgfc import providers as _implementation

sys.modules[__name__] = _implementation
