"""``python -m train``: run the command line in :mod:`train.api.cli`."""

import sys

from train.api.cli import main

sys.exit(main(sys.argv[1:]))
