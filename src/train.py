"""Train the GAM — config-driven (Hydra), logged (loguru + MLflow).

Every run records scalar metrics, per-term p-values, and 2 artifacts:
a partial-effects plot (with CI% bands) and residual
diagnostics — all visible under the run's Artifacts tab in the MLflow UI.

Run:
  # experiment — logs a run, nothing registered
  python -m src.train
  python -m src.train model=age_spline_bmi_spline
  python -m src.train gridsearch.lam_num=5 data.path=data/no_drift.csv

  # compare several models in one sweep
  python -m src.train -m model=age_spline_bmi_linear,age_spline_bmi_spline

  # stage a candidate — registers as @challenger, leaves @champion untouched
  python -m src.train registry.enabled=true model=age_spline_bmi_spline
  python -m src.train registry.enabled=true model=age_wiggly_spline_bmi_linear

  # promote the winner — moves @champion to this version (what serving loads)
  python -m src.train registry.enabled=true registry.promote=true

  # start MLflow
  mlflow server --backend-store-uri sqlite:///mlflow.db --port 5555 --workers 1

"""

import tempfile
from pathlib import Path

import cloudpickle
import hydra
import matplotlib

matplotlib.use("Agg")  # must run before importing pyplot — headless backend for Docker/CI

import matplotlib.pyplot as plt  # noqa: E402
import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pygam import LinearGAM, f, l, s
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

from src.helper_functions import plot_partial_effects, plot_residuals

TERM_BUILDERS = {
    "s": lambda t: s(t.col, n_splines=t.get("n_splines", 20)),
    "l": lambda t: l(t.col),
    "f": lambda t: f(t.col),
}


# this is a helper function to build our GAM terms from the config file, which is a list of dicts
def build_terms(term_cfgs):
    """Turn the YAML term list into a pygam terms object: s(0, n_splines=20) + l(1) + f(2)."""
    terms = TERM_BUILDERS[term_cfgs[0].kind](term_cfgs[0])
    for t in term_cfgs[1:]:
        terms += TERM_BUILDERS[t.kind](t)
    return terms


# this function uses mapping from the config file to encode categorical features into numeric val.
# returns the feature matrix X, target vector y, and feature names list
# this is used in the main training function to prepare the data for the GAM model
def load_xy(cfg):
    df = pd.read_csv(cfg.data.path)
    X = df.drop(columns=cfg.data.target)
    for col, mapping in cfg.schema.encoders.items():
        X[col] = X[col].map(mapping)
    return X.to_numpy(), df[cfg.data.target].to_numpy(), list(X.columns)


# Here are the metrics that will be logged to MLflow
def compute_metrics(gam, y_true, y_pred):
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 3),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 1),
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 1),
        "aic": round(float(gam.statistics_["AIC"]), 1),
        "edof": round(float(gam.statistics_["edof"]), 2),
        "pseudo_r2": round(float(gam.statistics_["pseudo_r2"]["explained_deviance"]), 3),
        "deviance": round(float(gam.statistics_["deviance"]), 1),
        "correlation": round(float(np.corrcoef(y_true, y_pred)[0, 1]), 3),
    }


# This chunk of code is the MLflow pyfunc wrapper for this project's (GAM) model,
# which allows you to serve the model behind the raw feature schema.
# It handles loading the model and making predictions based on the input data.
##---------------
# @champion model for the insurance-cost GAM.
##---------------
class GamModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.gam = cloudpickle.load(open(context.artifacts["gam"], "rb"))

    def predict(self, context, model_input, params=None):
        return self.gam.predict(np.asarray(model_input))


def log_and_register(gam, cfg):
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "gam.pkl"
        with open(path, "wb") as fh:
            cloudpickle.dump(gam, fh)
        info = mlflow.pyfunc.log_model(
            name="model", python_model=GamModel(), artifacts={"gam": str(path)}
        )
    if not cfg.registry.enabled:
        return
    name = cfg.registry.name
    mv = mlflow.register_model(info.model_uri, name)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(name, "challenger", mv.version)
    if cfg.registry.promote or int(mv.version) == 1:
        client.set_registered_model_alias(name, "champion", mv.version)
        logger.success("promoted {} v{} -> @champion", name, mv.version)
    else:
        logger.info("registered {} v{} -> @challenger", name, mv.version)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))

    X, y, feature_names = load_xy(cfg)
    logger.info("Loaded {} rows, features={}", len(y), feature_names)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=cfg.data.test_size, random_state=cfg.seed)
    # the user can specify the model terms in the config file,
    # e.g., LinearGAM(s(0) + l(1) + f(2))
    gam = LinearGAM(build_terms(cfg.model.terms))
    if cfg.gridsearch.enabled:
        lam = np.logspace(cfg.gridsearch.lam_min, cfg.gridsearch.lam_max, cfg.gridsearch.lam_num)
        logger.debug("gridsearch over {} lam values", cfg.gridsearch.lam_num)
        gam.gridsearch(Xtr, ytr, lam=lam, progress=False)
    else:
        gam.fit(Xtr, ytr)

    metrics = compute_metrics(gam, yva, gam.predict(Xva))
    logger.info("Metrics: {}", metrics)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)
    with mlflow.start_run(run_name=cfg.model.name):
        mlflow.log_params(
            {
                "model": cfg.model.name,
                "test_size": cfg.data.test_size,
                "gridsearch": cfg.gridsearch.enabled,
                "lam_num": cfg.gridsearch.lam_num,
                "data": cfg.data.path,
                "seed": cfg.seed,
                "CI": cfg.ci,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_text(OmegaConf.to_yaml(cfg), "config.yaml")

        # per-term p-values (scalars, one metric each)
        for name, p in zip(feature_names + ["intercept"], gam.statistics_["p_values"], strict=True):
            mlflow.log_metric(f"pval_{name}", float(p))

        # artifacts: partial-effects plot (PEP) as well as residual diagnostics
        # PEPs with CI confidence bands (conf. via Hydra; default = 95%)
        fig_pe = plot_partial_effects(gam, feature_names, cfg.ci)
        mlflow.log_figure(fig_pe, "partial_effects.png")
        plt.close(fig_pe)
        # residual diagnostics; in order to inspect the residuals,
        # we plot them against the fitted values and each feature
        fig_res = plot_residuals(gam, X, y, feature_names)
        mlflow.log_figure(fig_res, "residuals.png")
        plt.close(fig_res)
        # register the model if requested (and optionally promote to @champion)
        log_and_register(gam, cfg)

    logger.success("{} -> {}", cfg.model.name, metrics)


if __name__ == "__main__":
    main()
