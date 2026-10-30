"""``python -m train``: run the command line in :mod:`train.api.cli`."""

import sys

from train.api.cli import main
from train.api.recipes import register_builtin_models

register_builtin_models()
sys.exit(main(sys.argv[1:]))
