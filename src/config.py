import os

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

if not OmegaConf.has_resolver("cwd"):
    OmegaConf.register_new_resolver("cwd", lambda: os.getcwd().replace(os.sep, "/"))


def load_config(config_name: str = "config"):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../conf"):
        return compose(config_name=config_name)


cfg = load_config()

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", cfg.serve.mlflow.tracking_uri) # this is sqlite
# MODEL_URI is the path to the champion model in the MLflow registry
MODEL_URI = os.getenv("MODEL_URI", cfg.serve.mlflow.model_uri)
REGISTRY_NAME = cfg.serve.mlflow.registry_name
JSD_THRESHOLD = float(cfg.serve.monitor.jsd_threshold)
NUM_COLS = list(cfg.schema.num_cols)
CAT_COLS = list(cfg.schema.cat_cols)
TARGET = str(cfg.schema.target)
# you can check out:  python -c "from src.config import ENCODERS;; print(ENCODERS;)"; {'smoker': {'no': 0, 'yes': 1}}; 
# easy to use in the monitor.py and serve.py code for encoding categorical features
ENCODERS = OmegaConf.to_container(cfg.schema.encoders)
ALPHA = float(cfg.serve.monitor.alpha)
BINS = int(cfg.serve.monitor.bins)