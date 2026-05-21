import hydra
import pickle
from typing import Any

def cache_results(filename, object: Any) -> None:
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    out_dir = hydra_cfg["runtime"]["output_dir"]

    with open(f"{out_dir}/{filename}.pkl", "wb") as f:
        pickle.dump(object.cpu() if hasattr(object, "cpu") else object, f)