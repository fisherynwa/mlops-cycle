# run python -m pytest tests/test_train.py -v

from omegaconf import OmegaConf
from pygam.terms import SplineTerm, LinearTerm, FactorTerm

from src.train import build_terms


def cfg(term_list):
    """Wrap a list of dicts as OmegaConf, the way Hydra passes cfg.model.terms."""
    return OmegaConf.create(term_list)


# the standard 3-term spec used across most tests: s(0) + l(1) + f(2)
FULL_SPEC = [
    {"kind": "s", "col": 0, "n_splines": 15},
    {"kind": "l", "col": 1},
    {"kind": "f", "col": 2},
]


class TestBuildTerms:
    def test_builds_correct_number_of_terms(self):
        terms = build_terms(cfg(FULL_SPEC))
        assert len(terms) == 3

    def test_term_types_match_config(self):
        terms = build_terms(cfg(FULL_SPEC))
        assert isinstance(terms[0], SplineTerm)   # s -> spline
        assert isinstance(terms[1], LinearTerm)   # l -> linear
        assert isinstance(terms[2], FactorTerm)   # f -> factor

    def test_features_map_to_columns(self):
        terms = build_terms(cfg(FULL_SPEC))
        assert [t.feature for t in terms] == [0, 1, 2]

    def test_spline_uses_configured_n_splines(self):
        terms = build_terms(cfg([{"kind": "s", "col": 0, "n_splines": 15},
                                 {"kind": "l", "col": 1}]))
        assert terms[0].n_splines == 15

    def test_spline_defaults_n_splines_to_20(self):
        terms = build_terms(cfg([{"kind": "s", "col": 0},
                                 {"kind": "l", "col": 1}]))
        assert terms[0].n_splines == 20