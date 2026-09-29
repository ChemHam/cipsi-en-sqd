"""Per-molecule geometry.

Each module gives SPIN, COORDINATE and geometry(x, cfg).
"""

import importlib


def load(name):
    return importlib.import_module(f"systems.{name}")
