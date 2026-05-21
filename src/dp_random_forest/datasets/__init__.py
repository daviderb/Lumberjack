from .adult import get_adult_dataset
"""from .banknote import get_banknote_dataset
from .cardiovascular import get_cardiovascular_dataset"""
from .folktables import get_folktables_dataset
from .moons import get_moons
from .unbalanced_moons import get_unbalanced_moons
"""from .mushroom import get_mushroom_dataset
from .nursery import get_nursery_dataset
from .synthetic import get_synthetic_dataset
from .synthetic_categorical import get_synthetic_categorical_dataset
from .synthetic_categorical_parity import get_synthetic_categorical_parity_dataset
from .vehicle import get_vehicle_dataset
from .wbc import get_wbc_dataset"""


def get_dataset(cfg):
    """if cfg.dataset.name == "synthetic":
        return get_synthetic_dataset(cfg)
    elif cfg.dataset.name == "synthetic_categorical":
        return get_synthetic_categorical_dataset(cfg)
    elif cfg.dataset.name == "synthetic_categorical_parity":
        return get_synthetic_categorical_parity_dataset(cfg)
    elif cfg.dataset.name == "banknote":
        return get_banknote_dataset(cfg)
    elif cfg.dataset.name == "mushroom":
        return get_mushroom_dataset(cfg)
    elif cfg.dataset.name == "nursery":
        return get_nursery_dataset(cfg)
    elif cfg.dataset.name == "cardiovascular":
        return get_cardiovascular_dataset(cfg)
    elif cfg.dataset.name == "vehicle":
        return get_vehicle_dataset(cfg)
    elif cfg.dataset.name == "wbc":
        return get_wbc_dataset(cfg)"""
    if cfg.dataset.name == "moons":
        return get_moons(cfg)
    elif cfg.dataset.name == "moonsunbalanced":
        return get_unbalanced_moons(cfg)
    elif cfg.dataset.name == "adult":
        return get_adult_dataset(cfg)
    elif cfg.dataset.name in ["folktables", "folktables-income", "folktables-employment", "folktables-coverage", "folktables-travel", "folktables-full"]:
        return get_folktables_dataset(cfg)
    else:
        raise ValueError(f"Dataset {cfg.dataset.name} not found")
