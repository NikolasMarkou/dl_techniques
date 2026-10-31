"""``python -m train``: run the command line in :mod:`train.api.cli`.

The built-in families and tasks are registered by ``main`` itself, so this module
is only the unconditional exit.
"""

import sys

from train.api.cli import main

sys.exit(main(sys.argv[1:]))
