import importlib.util as ilu
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = ilu.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    mod = ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
