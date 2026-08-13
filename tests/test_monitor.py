"""Unit tests for monitor.score — encoding, column order, prediction, no mutation.

Run:

python -m pytest tests/test_monitor.py -v
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import anderson_ksamp, ks_2samp

from src.config import CAT_COLS, ENCODERS, NUM_COLS
from src.helper_functions import ad_test, ecdf_plot, proptest  # adjust import to your layout
from src.monitor import score

ALPHA = 0.05



class FakeModel:
    """Stand-in for the champion: records its input, returns a fixed prediction."""

    def __init__(self):
        self.seen = None

    def predict(self, X):
        self.seen = X.copy()
        return np.arange(len(X), dtype=float)


@pytest.fixture
def raw_df():
    return pd.DataFrame(
        {
            "age": [25, 40, 60],
            "bmi": [22.0, 30.5, 28.1],
            "smoker": ["no", "yes", "no"],
            "charges": [5000.0, 30000.0, 12000.0],
        }
    )


class TestScore:
    """Tests for the score() function: encoding, ordering, prediction, no mutation."""

    def test_adds_prediction_column(self, raw_df):
        model = FakeModel()
        out = score(raw_df, model)
        assert "prediction" in out.columns

    def test_smoker_is_encoded_before_predict(self, raw_df):
        model = FakeModel()
        score(raw_df, model)
        # the model should have received encoded smoker (0/1), not strings
        assert model.seen["smoker"].tolist() == [
            ENCODERS["smoker"]["no"],
            ENCODERS["smoker"]["yes"],
            ENCODERS["smoker"]["no"],
        ]
        assert model.seen["smoker"].dtype != object

    def test_column_order_matches_training(self, raw_df):
        model = FakeModel()
        score(raw_df, model)
        assert list(model.seen.columns) == NUM_COLS + CAT_COLS

    def test_original_columns_preserved(self, raw_df):
        out = score(raw_df, FakeModel())
        for col in raw_df.columns:
            assert col in out.columns  # charges, age, bmi, smoker all still there


### Test ecdf_plot() in helper_functions.py, which is used by monitor.score()

def test_returns_python_float(tmp_path):
    d = ecdf_plot([1.0, 2, 3], [2.0, 3, 4], "x", tmp_path / "e.png")
    assert isinstance(d, float)          # float(D), not np.float64 -> JSON-safe for MLflow
    assert 0.0 <= d <= 1.0


def test_writes_file(tmp_path):
    p = tmp_path / "ecdf.png"
    ecdf_plot([1.0, 2, 3], [1.0, 2, 3], "x", p)
    assert p.exists() and p.stat().st_size > 0


def test_identical_samples_zero_distance(tmp_path):
    d = ecdf_plot([1.0, 2, 3, 4], [1.0, 2, 3, 4], "x", tmp_path / "e.png")
    assert d == 0.0                      # no gap between identical ECDFs


def test_disjoint_samples_max_distance(tmp_path):
    d = ecdf_plot([1.0, 2, 3], [10.0, 11, 12], "x", tmp_path / "e.png")
    assert d == pytest.approx(1.0)       # fully separated -> D = 1


@pytest.mark.parametrize("seed", range(5))
def test_matches_scipy_ks_2samp(tmp_path, seed):
    rng = np.random.default_rng(seed)
    ref = rng.normal(0, 1, 200)
    cur = rng.normal(0.4, 1, 150)        # unequal sizes on purpose
    d = ecdf_plot(ref, cur, "x", tmp_path / "e.png")
    assert d == pytest.approx(ks_2samp(ref, cur).statistic, abs=1e-9)


def test_unequal_lengths_ok(tmp_path):
    d = ecdf_plot(np.arange(1000.0), np.arange(3.0), "x", tmp_path / "e.png")
    assert 0.0 <= d <= 1.0


#####################################
##  Compare KS D from ecdf_plot vs scipy.stats.ks_2samp across many cases."""
#####################################
CASES = {
    "identical":      (np.arange(1, 11.0),            np.arange(1, 11.0)),
    "disjoint":       (np.arange(1, 11.0),            np.arange(20, 30.0)),
    "shift":          (np.arange(1, 11.0),            np.arange(4, 14.0)),
    "unequal_sizes":  (np.arange(1, 101.0),           np.arange(1, 31.0)),
    "heavy_ties":     (np.array([1, 1, 1, 2, 2, 3.]), np.array([2, 2, 3, 3, 3, 4.])),
    "one_off_shift":  (np.arange(1, 11.0),            np.arange(2, 12.0)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_D_matches_scipy_fixed_cases(tmp_path, name):
    ref, cur = CASES[name]
    project = ecdf_plot(ref, cur, name, tmp_path / f"{name}.png")
    scipy_impl = ks_2samp(ref, cur).statistic
    assert project == pytest.approx(scipy_impl, abs=1e-12), f"{name}: {project} vs {scipy_impl}"


@pytest.mark.parametrize("seed", range(20))
def test_D_matches_scipy_random(tmp_path, seed):
    rng = np.random.default_rng(seed)
    n, m = rng.integers(30, 400), rng.integers(30, 400)
    ref = rng.normal(0, 1, n)
    cur = rng.normal(rng.uniform(-1, 1), rng.uniform(0.5, 2), m)  # random loc + scale
    project = ecdf_plot(ref, cur, "x", tmp_path / f"{seed}.png")
    scipy_impl = ks_2samp(ref, cur).statistic
    assert project == pytest.approx(scipy_impl, abs=1e-12), f"{seed}: {project} vs {scipy_impl}"


def test_D_matches_scipy_with_ties(tmp_path):
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 10, 500).astype(float)   # integer -> lots of ties
    cur = rng.integers(2, 12, 500).astype(float)
    project = ecdf_plot(ref, cur, "x", tmp_path / "ties.png")
    scipy_impl = ks_2samp(ref, cur).statistic
    assert project == pytest.approx(scipy_impl, abs=1e-12), f"ties: {project} vs {scipy_impl}"


##########################
## Test ad_test() and proptest() in helper_functions.py
#########################
# ----------------------------- ad_test -----------------------------
def test_ad_keys_and_types():
    rng = np.random.default_rng(0)
    r = ad_test(rng.normal(size=200), rng.normal(0.5, size=200), ALPHA)
    assert set(r) == {"test", "statistic", "p_value", "shift", "significant"}
    assert isinstance(r["statistic"], float) and isinstance(r["p_value"], float)
    assert isinstance(r["shift"], float) and isinstance(r["significant"], bool)
    assert r["test"] == "anderson_darling"

def test_ad_no_drift_not_significant():
    rng = np.random.default_rng(7)
    r = ad_test(rng.normal(size=500), rng.normal(size=500), ALPHA)  # same distribution
    assert r["significant"] is False
    assert r["p_value"] > ALPHA

def test_ad_clear_drift_significant():
    rng = np.random.default_rng(1)
    r = ad_test(rng.normal(0, 1, 400), rng.normal(3, 1, 400), ALPHA)
    assert r["significant"] is True

def test_ad_shift_sign_matches_direction():
    rng = np.random.default_rng(2)
    up = ad_test(rng.normal(0, 1, 300), rng.normal(2, 1, 300), ALPHA)
    down = ad_test(rng.normal(0, 1, 300), rng.normal(-2, 1, 300), ALPHA)
    assert up["shift"] > 0 and down["shift"] < 0

def test_ad_statistic_matches_scipy():
    rng = np.random.default_rng(3)
    ref, cur = rng.normal(size=150), rng.normal(0.4, size=120)  # unequal sizes
    got = ad_test(ref, cur, ALPHA)["statistic"]
    assert got == pytest.approx(anderson_ksamp([ref, cur]).statistic, abs=1e-9)

# ----------------------------- proptest ----------------------------
def test_prop_keys_and_types():
    ref = pd.Series(["yes"]*20 + ["no"]*80)
    cur = pd.Series(["yes"]*40 + ["no"]*60)
    r = proptest(ref, cur, "yes", ALPHA)
    assert set(r) == {"test", "ref_rate", "cur_rate", "rate_shift", "z", "p_value", "significant"}
    assert isinstance(r["p_value"], float) and isinstance(r["significant"], bool)

def test_prop_rates_and_shift():
    ref = pd.Series(["yes"]*20 + ["no"]*80)   # 20%
    cur = pd.Series(["yes"]*45 + ["no"]*55)   # 45%
    r = proptest(ref, cur, "yes", ALPHA)
    assert r["ref_rate"] == pytest.approx(0.20)
    assert r["cur_rate"] == pytest.approx(0.45)
    assert r["rate_shift"] == pytest.approx(0.25)

def test_prop_no_change_not_significant():
    ref = pd.Series(["yes"]*300 + ["no"]*700)
    cur = pd.Series(["yes"]*300 + ["no"]*700)  # identical rates
    assert proptest(ref, cur, "yes", ALPHA)["significant"] is False

def test_prop_big_change_significant():
    ref = pd.Series(["yes"]*200 + ["no"]*800)  # 20%
    cur = pd.Series(["yes"]*500 + ["no"]*500)  # 50%
    assert proptest(ref, cur, "yes", ALPHA)["significant"] is True

# --------------------- feat_tests assembly -------------------------
def _feat_tests(ref_raw, cur_raw, positive, alpha):
    return {
        "age": ad_test(ref_raw["age"].to_numpy(), cur_raw["age"].to_numpy(), alpha),
        "smoker": proptest(ref_raw["smoker"], cur_raw["smoker"], positive, alpha),
        "bmi": ad_test(ref_raw["bmi"].to_numpy(), cur_raw["bmi"].to_numpy(), alpha),
    }

def test_feat_tests_structure_and_drift_sources():
    rng = np.random.default_rng(5)
    ref = pd.DataFrame({"age": rng.integers(20, 60, 300).astype(float),
                        "bmi": rng.normal(28, 5, 300),
                        "smoker": ["yes"]*60 + ["no"]*240})
    cur = pd.DataFrame({"age": rng.integers(30, 70, 300).astype(float),  # shifted up
                        "bmi": rng.normal(31, 5, 300),                   # shifted up
                        "smoker": ["yes"]*150 + ["no"]*150})             # shifted up
    ft = _feat_tests(ref, cur, "yes", ALPHA)
    assert set(ft) == {"age", "smoker", "bmi"}
    assert all("significant" in v for v in ft.values())   # drift_sources relies on this
    drift_sources = [f for f, r in ft.items() if r["significant"]]
    assert drift_sources == ["age", "smoker", "bmi"]