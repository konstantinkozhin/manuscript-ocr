"""Replace optional modules without rolling back unrelated imports."""
from contextlib import contextmanager
import sys


@contextmanager
def patch_modules(values):
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in values}
    sys.modules.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
