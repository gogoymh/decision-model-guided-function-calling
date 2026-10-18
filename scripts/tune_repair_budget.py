"""Choose the shared repair budget from saved JBFC development traces.

Not part of the paper protocol since 2026-10-18: the paper fixes r=2 and runs every
example once (make table1, make table2). Kept for optional exploration only.

The test repair methods use this same limit; this command does not select a
separate limit for each method.
"""

from jev_fc.tuning import main


if __name__ == "__main__":
    raise SystemExit(main())
