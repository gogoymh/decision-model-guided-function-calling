"""DM-Call name for :mod:`dmgfc.tools`; both names refer to the same module."""

import sys

from dmgfc import tools as _implementation

sys.modules[__name__] = _implementation
