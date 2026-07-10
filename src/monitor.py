"""Monitor a production batch for drift against the reference set.

Scores both the reference and the current batch with the @champion model, then
builds an Evidently report using Jensen-Shannon distance for numeric drift.
Saves a shareable HTML report (the thing you record for the GIF) and prints a
machine-readable verdict.

Run:
  python -m src.monitor                              # default: data_drift.csv
  python -m src.monitor --current data/no_drift.csv  # the quiet control

  mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5500 --workers 1

  $env:MLFLOW_TRACKING_URI = "http://localhost:5555"
  python -m src.monitor --current data/data_drift.csv --out reports/drift_report.html
"""

import argparse
import json
import os
from pathlib import Path

import mlflow
import pandas as pd
from evidently import DataDefinition, Dataset, Regression, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset, RegressionPreset
from loguru import logger
from src.config import (MLFLOW_URI, MODEL_URI, JSD_THRESHOLD, NUM_COLS, CAT_COLS, TARGET, ENCODERS, ALPHA, BINS)
from scipy.spatial.distance import jensenshannon
import numpy as np
from src.helper_functions import js_distance, ks, proptest

# --- the score() function is tested in tests/test_monitor.py ---
def score(df: pd.DataFrame, model) -> pd.DataFrame:
    """Add champion predictions so we can monitor prediction/target drift too."""
    out = df.copy()
    enc = df.copy()
    for col, mapping in ENCODERS.items():
        enc[col] = enc[col].map(mapping)
    out["prediction"] = model.predict(enc[NUM_COLS + CAT_COLS]) # model is the champion, already loaded from MLflow
    return out


def to_dataset(df: pd.DataFrame) -> Dataset:
    definition = DataDefinition(
        numerical_columns=NUM_COLS, # numeric columns are the features, not the target; evidently requires this to be explicit
        categorical_columns=CAT_COLS, # categorical columns are the features, not the target; evidently requires this to be explicit
        regression=[Regression(target=TARGET, prediction="prediction")],
    )
    return Dataset.from_pandas(df, data_definition=definition)


def monitor(reference_csv: str, current_csv: str, out_html: str) -> dict:
    mlflow.set_tracking_uri(MLFLOW_URI)
    # load the champion model from the MLflow registry (or local path)
    model = mlflow.pyfunc.load_model(MODEL_URI)
    # score both the reference and current batches with the champion model
    ref = score(pd.read_csv(reference_csv), model) 
    cur = score(pd.read_csv(current_csv), model)
    logger.info("Scored {} reference / {} current rows", len(ref), len(cur))

    report = Report(
        metrics=[
            DataDriftPreset(num_method="jensenshannon", num_threshold=JSD_THRESHOLD),
            RegressionPreset(),
            DataSummaryPreset(),
        ],
        include_tests=True,
    )
    snapshot = report.run(to_dataset(cur), to_dataset(ref))

    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(out_html)
    logger.success("Report saved -> {}", out_html)

    jsd = {c: round(js_distance(ref[c].to_numpy(), cur[c].to_numpy()), 3)
           for c in NUM_COLS + [TARGET]}

    # feature-level attribution (statsmodels tests) on the raw batches
    ref_raw = pd.read_csv(reference_csv) # train data
    cur_raw = pd.read_csv(current_csv)
    positive = next(k for k, v in ENCODERS["smoker"].items() if v == 1)
    feat_tests = {
        "age": ks(ref_raw["age"].to_numpy(), cur_raw["age"].to_numpy(), ALPHA),
        "smoker": proptest(ref_raw["smoker"], cur_raw["smoker"], positive, ALPHA),
        "bmi": ks(ref_raw["bmi"].to_numpy(), cur_raw["bmi"].to_numpy(), ALPHA),
    }
    drift_sources = [f for f, r in feat_tests.items() if r["significant"]]

    summary = {
        "batch": Path(current_csv).stem,
        "jsd": jsd,
        "drift_detected": any(v > JSD_THRESHOLD for v in jsd.values()), # if any drift is detected, we flag the batch as drifted
        "feature_tests": feat_tests,
        "drift_sources": drift_sources,
        "report": out_html,
    }
    logger.info("Summary: {}", json.dumps(summary))

    # log Evidently report + JSD + feature tests to one MLflow run
    mlflow.set_experiment("drift_monitoring")
    with mlflow.start_run(run_name=summary["batch"]):
        mlflow.log_params({"batch": summary["batch"], "reference": reference_csv,
                           "current": current_csv, "alpha": ALPHA})
        mlflow.log_metrics({f"jsd_{c}": v for c, v in jsd.items()})
        mlflow.log_metric("drift_detected", int(summary["drift_detected"]))
        mlflow.log_metric("age_shift", feat_tests["age"]["shift"])
        mlflow.log_metric("age_ks_pvalue", feat_tests["age"]["p_value"])
        mlflow.log_metric("smoker_rate_shift", feat_tests["smoker"]["rate_shift"])
        mlflow.log_metric("smoker_pvalue", feat_tests["smoker"]["p_value"])
        mlflow.log_metric("bmi_shift", feat_tests["bmi"]["shift"])
        mlflow.log_metric("bmi_ks_pvalue", feat_tests["bmi"]["p_value"])
        mlflow.log_artifact(out_html)                         # the Evidently HTML
        mlflow.log_text(json.dumps(feat_tests, indent=2), "feature_tests.json")
    logger.success("Logged report + feature tests to MLflow (sources: {})",
                   drift_sources or "none")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reference", default="data/reference.csv")
    p.add_argument("--current", default="data/data_drift.csv")
    p.add_argument("--out", default="reports/drift_report.html")
    args = p.parse_args()
    monitor(args.reference, args.current, args.out)