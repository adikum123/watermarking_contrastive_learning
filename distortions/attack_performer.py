import importlib.util
import inspect
import sys
from pathlib import Path

from distortions.attacks.attacks.base_attack import BaseAttack


class AttackPerformer:

    def __init__(self, attacks_dir="distortions/attacks/attacks"):
        self.attacks_dir = Path(attacks_dir)
        self.attack_classes = self.load_classes()

    def load_classes(self):
        classes = {}
        for attack_subdir in self.attacks_dir.iterdir():
            if attack_subdir.is_dir():
                for py_file in attack_subdir.glob("*.py"):
                    # Use full relative path as module name to avoid collisions
                    relative_path = py_file.relative_to(self.attacks_dir.parent)
                    module_name = ".".join(relative_path.with_suffix("").parts)

                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module  # register module
                    spec.loader.exec_module(module)

                    # Grab all classes defined in the module
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, BaseAttack) and obj is not BaseAttack:
                            classes[name] = obj
        return classes

    def instantiate_all(self, *args, **kwargs):
        instances = {}
        for name, cls in self.attack_classes.items():
            instances[name] = cls(*args, **kwargs)
        return instances


if __name__ == "__main__":
    # Usage
    loader = AttackPerformer()
    print(loader.attack_classes)