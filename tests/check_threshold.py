"""
Moved.

This file used to run at import time: `pytest` collecting the tests/
directory would load the embedding model, open the vector store, and print
a table to stdout - before a single test ran. It is a diagnostic tool, not a
test, so it now lives in scripts/.

    python scripts/check_threshold.py

This shim is kept only so an old command line fails with a clear message
instead of an ImportError.
"""

import sys

MESSAGE = (
    "check_threshold.py has moved. Run it as:\n"
    "    python scripts/check_threshold.py\n"
)


if __name__ == "__main__":
    sys.exit(MESSAGE)
