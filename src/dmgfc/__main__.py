"""Support ``python -m dmgfc`` as the DMGFC evaluation command."""

import sys

from .cli import main

sys.exit(main())
