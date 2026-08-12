"""Monitor a production batch for drift against the reference set.

Run:
  python -m src.monitor                              # default: data_drift.csv
  python -m src.monitor --current data/no_drift.csv  # the quiet control
  python -m src.monitor --current data/data_drift.csv --out reports/drift_report.html

MLflow tracking URI (via MLFLOW_TRACKING_URI env var or cfg.serve.mlflow.tracking_uri):
  * docker-compose: server listens on 5000 in-container, published as "5555:5000".
      - from the host:                 http://localhost:5555
      - from another compose service:  http://mlflow:5000   (service name + internal port)
  * local (no docker):
      mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
      $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
"""

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd
from evidently import DataDefinition, Dataset, Regression, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset, RegressionPreset

import sys
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO")
from src.config import (
    ALPHA,
    CAT_COLS,
    ENCODERS,
    MLFLOW_URI,
    MODEL_URI,
    NUM_COLS,
    TARGET,
    WASSERSTEIN_STD_THRESHOLD,  # standardized threshold for the Evidently report only
)
from src.helper_functions import ad_test, ecdf_plot, proptest, wasserstein_dist


# --- the score() function is tested in tests/test_monitor.py ---
def score(df: pd.DataFrame, model) -> pd.DataFrame:
    """Add champion predictions so we can monitor prediction/target drift too."""
    out = df.copy()
    enc = df.copy()
    for col, mapping in ENCODERS.items():
        enc[col] = enc[col].map(mapping)
    out["prediction"] = model.predict(enc[NUM_COLS + CAT_COLS])  # load @champion
    return out


def to_dataset(df: pd.DataFrame) -> Dataset:
    definition = DataDefinition(
        numerical_columns=NUM_COLS,
        categorical_columns=CAT_COLS,
        regression=[Regression(target=TARGET, prediction="prediction")],
    )
    return Dataset.from_pandas(df, data_definition=definition)


# Evidently defaults (as of 2026 docs):
#   reference size   numerical (n_unique>5)          categorical / low-cardinality
#   <= 1000 rows     two-sample KS (p<0.05)          Chi-square; binary -> Z-test
#   >  1000 rows     Wasserstein (>=0.1)             Jensen-Shannon (>=0.1)
# NOTE: Evidently's "wasserstein" is STANDARDIZED (distance / reference SD), so its
# 0.1-style threshold is unitless. That is intentionally the *standardized* threshold for the Evidently report only. 
# The raw (unstandardized) Wasserstein is reported in the summary and logged to MLflow, 
# so it can be compared against a PER-COLUMN threshold in the column's own units.
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
            # numerical columns by means of the Wasserstein distance (standardized inside Evidently)
            DataDriftPreset(num_method="wasserstein", num_threshold=WASSERSTEIN_STD_THRESHOLD),
            RegressionPreset(),
            DataSummaryPreset(),
        ],
        include_tests=True,
    )
    snapshot = report.run(to_dataset(cur), to_dataset(ref))

    out_dir = Path(out_html).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(out_html)
    logger.success("Report saved: {}", out_html)

    # drift metric: RAW (unstandardized) Wasserstein per numeric col 
    wass = {
        c: round(wasserstein_dist(ref[c].to_numpy(), cur[c].to_numpy()), 3)
        for c in NUM_COLS + [TARGET]
    }

    # feature-level attribution on the raw batches:
    #   numeric through Anderson-Darling (tail-sensitive; replaces KS)
    #   binary by means of proportion / Z-test
    ref_raw = pd.read_csv(reference_csv)
    cur_raw = pd.read_csv(current_csv)
    positive = next(k for k, v in ENCODERS["smoker"].items() if v == 1)
    feat_tests = {
        "age": ad_test(ref_raw["age"].to_numpy(), cur_raw["age"].to_numpy(), ALPHA),
        "smoker": proptest(ref_raw["smoker"], cur_raw["smoker"], positive, ALPHA),
        "bmi": ad_test(ref_raw["bmi"].to_numpy(), cur_raw["bmi"].to_numpy(), ALPHA),
    }
    drift_sources = [f for f, r in feat_tests.items() if r["significant"]]

    # ECDF diagnostics (the KS picture and its effect size D) for the numeric columns; these are artifacts in the report
    ks_d, ecdf_paths = {}, {}
    for col in NUM_COLS:
        path = str(out_dir / f"ecdf_{col}.png")
        ks_d[col] = round(
            ecdf_plot(ref_raw[col].to_numpy(), cur_raw[col].to_numpy(), col, path), 3
        )
        ecdf_paths[col] = path

    summary = {
        "batch": Path(current_csv).stem,
        "wasserstein": wass,  # raw distances, reported as magnitudes (non-zero => some drift)
        "ks_d": ks_d,         # KS statistic from the ECDF plots (diagnostic)
        # decision comes from the tests' p-values:
        # the AD test for numeric, proportion for binary);
        # the Wasserstein distance is reported, not a gate -> also covers categorical drift
        "drift_detected": bool(drift_sources),
        "feature_tests": feat_tests,
        "drift_sources": drift_sources,
        "report": out_html,
    }
    logger.info("Summary: {}", json.dumps(summary))

    # log Evidently report + metrics + feature tests + ECDFs to one MLflow run
    mlflow.set_experiment("drift_monitoring")
    with mlflow.start_run(run_name=summary["batch"]):
        mlflow.log_params(
            {
                "batch": summary["batch"],
                "reference": reference_csv,
                "current": current_csv,
                "alpha": ALPHA,
            }
        )
        # raw Wasserstein per numeric column (+ target)
        mlflow.log_metrics({f"wasserstein_{c}": v for c, v in wass.items()})
        mlflow.log_metric("drift_detected", int(summary["drift_detected"]))
        # Anderson-Darling (replaces KS) for the numeric features
        mlflow.log_metric("age_shift", feat_tests["age"]["shift"])
        mlflow.log_metric("age_ad_statistic", feat_tests["age"]["statistic"])
        mlflow.log_metric("age_ad_pvalue", feat_tests["age"]["p_value"])
        mlflow.log_metric("bmi_shift", feat_tests["bmi"]["shift"])
        mlflow.log_metric("bmi_ad_statistic", feat_tests["bmi"]["statistic"]) # track effect size too
        mlflow.log_metric("bmi_ad_pvalue", feat_tests["bmi"]["p_value"])
        # proportion test (kept) for the binary feature
        mlflow.log_metric("smoker_rate_shift", feat_tests["smoker"]["rate_shift"])
        mlflow.log_metric("smoker_pvalue", feat_tests["smoker"]["p_value"])
        # ECDF artifacts + their KS D (the KS diagnostic, per request)
        for col in NUM_COLS:
            mlflow.log_metric(f"{col}_ks_d", ks_d[col])
            mlflow.log_artifact(ecdf_paths[col])
        mlflow.log_artifact(out_html)  # the Evidently HTML
        mlflow.log_text(json.dumps(feat_tests, indent=2), "feature_tests.json")
    logger.success("Logged report + AD tests + ECDFs to MLflow (sources: {})", drift_sources or "none")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reference", default="data/reference.csv")
    p.add_argument("--current", default="data/data_drift.csv")
    p.add_argument("--out", default="reports/drift_report.html")
    args = p.parse_args()
    monitor(args.reference, args.current, args.out)