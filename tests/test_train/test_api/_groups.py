"""The argument groups of ``train.api.args``, found by looking, not by a list.

A hand-written stand-in for the task config (``Task`` with seven named groups)
cannot notice an eighth group class added to ``args.py``: the stand-in would
simply not carry it, and every test over the stand-in would stay green. These
two helpers derive the stand-in from the module, so a new group is in it the
moment it exists.

Interface contract
------------------
``args_groups()``
    :return: every :class:`~train.api.config.StrictConfig` subclass defined in
        ``train.api.args``, sorted by class name. Never empty.

``field_name(group)``
    :return: the snake_case task-config key for a group class, by the naming
        rule the groups follow (``OptimArgs`` is ``optim_args``).

``task_class()``
    :return: a frozen :class:`~train.api.config.StrictConfig` dataclass named
        ``Task`` with one field per group, each defaulting to ``group()``.
"""

import dataclasses
import inspect
import re
from typing import List, Type

import train.api.args as args_module
from train.api.config import StrictConfig


def args_groups() -> List[Type[StrictConfig]]:
    groups = [
        cls
        for _name, cls in inspect.getmembers(args_module, inspect.isclass)
        if issubclass(cls, StrictConfig)
        and cls is not StrictConfig
        and cls.__module__ == args_module.__name__
    ]
    assert groups, "train.api.args defines no group, so the derivation is broken"
    return groups


def field_name(group: Type[StrictConfig]) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", group.__name__).lower()


def task_class() -> Type[StrictConfig]:
    return dataclasses.make_dataclass(
        "Task",
        [
            (field_name(group), group, dataclasses.field(default_factory=group))
            for group in args_groups()
        ],
        bases=(StrictConfig,),
        frozen=True,
    )
