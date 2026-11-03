"""Support ``python -m dmcall`` as the DM-Call evaluation command."""

import sys

from .cli import main

sys.exit(main())
