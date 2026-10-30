"""``python -m train``: run the command line in :mod:`train.api.cli`."""

import sys

from train.api.cli import main
from train.api.recipes import register_builtin_models
from train.api.tasks import register_builtin_tasks

register_builtin_models()
register_builtin_tasks()
sys.exit(main(sys.argv[1:]))
