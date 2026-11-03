"""DM-Call name for :mod:`dmgfc._prompt`; both names refer to the same module."""

import sys

from dmgfc import _prompt as _implementation

sys.modules[__name__] = _implementation
