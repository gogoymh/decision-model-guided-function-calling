"""DM-Call name for :mod:`dmgfc.jev`; both names refer to the same module."""

import sys

from dmgfc import jev as _implementation

sys.modules[__name__] = _implementation
