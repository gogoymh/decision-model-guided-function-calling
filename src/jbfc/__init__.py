"""Compatibility facade for the former ``jbfc`` package.

Use ``import dmgfc`` and ``dmgfc.DMGFC`` for new code. ``DMFC`` and ``JBFC``
remain aliases of that same class; old method strings select the same repair loop.
"""

from dmgfc import *  # noqa: F403
from dmgfc import __all__, __version__  # noqa: F401
