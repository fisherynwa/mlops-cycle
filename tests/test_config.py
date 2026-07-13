from src import config as c


class TestConfig:
    def test_schema_columns_are_lists(self):
        assert isinstance(c.NUM_COLS, list)
        assert isinstance(c.CAT_COLS, list)
        assert c.NUM_COLS == ["age", "bmi"]
        assert c.CAT_COLS == ["smoker"]

    def test_target_is_charges(self):
        assert c.TARGET == "charges"

    def test_encoders_is_plain_dict(self):
        # OmegaConf.to_container should give real dicts, not DictConfig objects
        assert isinstance(c.ENCODERS, dict)

    def test_jsd_threshold_is_float(self):
        assert isinstance(c.JSD_THRESHOLD, float)
        assert 0 < c.JSD_THRESHOLD < 1

    def test_model_uri_points_at_champion(self):
        assert "@champion" in c.MODEL_URI
        assert c.REGISTRY_NAME in c.MODEL_URI

    def test_encoders_cover_categorical_columns(self):
        # every categorical column must have an encoder defined
        for col in c.CAT_COLS:
            assert col in c.ENCODERS
