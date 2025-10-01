import importlib.util
import inspect
import random
import sys
from collections import defaultdict
from pathlib import Path

from distortions.attacks.attacks.base_attack import BaseAttack


class AttackPerformer:

    def __init__(self, attacks_dir="distortions/attacks/attacks"):
        self.attacks_dir = Path(attacks_dir)
        self.attack_classes = self.load_classes()

        # group attacks by type
        self.by_type = defaultdict(list)
        for _, cls_object in self.attack_classes.items():
            self.by_type[cls_object.type].append(cls_object)

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
                            classes[name] = obj()
        return classes

    def instantiate_all(self, *args, **kwargs):
        instances = {}
        for name, cls in self.attack_classes.items():
            instances[name] = cls(*args, **kwargs)
        return instances

    def get_contrastive_views(self, x, sr):
        for attack_type in ["spectral", "structural"]:
            x = random.choice(self.by_type[attack_type]).apply(x, sampling_rate=sr)
        return x


if __name__ == "__main__":
    # Usage
    loader = AttackPerformer()
    print(loader.by_type)