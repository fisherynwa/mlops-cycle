"""Unit tests for monitor.score — encoding, column order, prediction, no mutation.

Uses a fake model that records the DataFrame it receives, so we can assert the
encoding and column order are correct without needing a real MLflow champion.

Run:

python -m pytest tests/test_monitor.py -v
"""
"""Unit tests for monitor.score, grouped in a class."""

import numpy as np
import pandas as pd
import pytest

from src.monitor import score
from src.config import ENCODERS, NUM_COLS, CAT_COLS
from src.helper_functions import ks_test, proptest


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


class TestKS:
    """Tests for the ks_test() function: flags shifted data, quiet on same distribution."""
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.ref = rng.normal(40, 9, 1000)

    def test_ks_flags_shifted_data(self):
        cur = self.ref + 25                     # clearly shifted
        result = ks_test(self.ref, cur, alpha=0.05)
        assert result["significant"] is True

    def test_ks_quiet_on_same_distribution(self):
        rng = np.random.default_rng(1)
        cur = rng.normal(40, 9, 1000)      # same distribution
        result = ks_test(self.ref, cur, alpha=0.01)
        assert result["significant"] is False