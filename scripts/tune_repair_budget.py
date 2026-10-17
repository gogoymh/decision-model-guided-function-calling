"""Choose the shared repair budget from saved JBFC development traces.

The test repair methods use this same limit; this command does not select a
separate limit for each method.
"""

from jev_fc.tuning import main


if __name__ == "__main__":
    raise SystemExit(main())
