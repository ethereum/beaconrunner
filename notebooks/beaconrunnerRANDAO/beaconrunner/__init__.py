__all__ = ['network', 'simulator', 'specs', 'utils', 'validatorlib', 'validators']

import os
import types
import importlib

from . import (
    network,
    simulator,
    specs,
    utils,
    validatorlib,
    validators,
)

def reload_package(package):
    # from https://stackoverflow.com/questions/28101895/reloading-packages-and-their-submodules-recursively-in-python
    assert(hasattr(package, "__package__"))
    fn = package.__file__
    fn_dir = os.path.dirname(fn) + os.sep
    module_visit = {fn}
    del fn

    def reload_recursive_ex(module):
        importlib.reload(module)

        for module_child in vars(module).values():
            if isinstance(module_child, types.ModuleType):
                fn_child = getattr(module_child, "__file__", None)
                if (fn_child is not None) and fn_child.startswith(fn_dir):
                    if fn_child not in module_visit:
                        # print("reloading:", fn_child, "from", module)
                        module_visit.add(fn_child)
                        reload_recursive_ex(module_child)

    return reload_recursive_ex(package)
