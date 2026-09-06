"""Gallery data domains, row selection, and natural-parameter interpretation."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import expit


@pytest.fixture
def gallery():
    path = Path(__file__).resolve().parents[1] / "examples" / "gallery_generalized.py"
    spec = importlib.util.spec_from_file_location("gallery_generalized", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_votes_complete_case_selection_ignores_party_and_records_columns(gallery, tmp_path):
    path = tmp_path / "votes.data"
    rows = [["y"] * 16, ["n"] * 15 + ["y"], ["?"] + ["n"] * 15]
    path.write_text("\n".join("partyA," + ",".join(row) for row in rows), encoding="utf-8")
    X, ids, columns = gallery.load_votes(path)
    assert_array_equal(ids, [0, 1])
    assert_array_equal(columns, np.arange(15))
    assert_array_equal(X, np.array([[1] * 15, [0] * 15]))
    path.write_text(path.read_text().replace("partyA", "differentParty"), encoding="utf-8")
    assert_array_equal(gallery.load_votes(path)[0], X)


def test_bike_counts_remain_raw_and_chain_is_chronological(gallery, tmp_path):
    path = tmp_path / "day.csv"
    path.write_text(
        "dteday,cnt\n2011-01-01,1234\n2011-01-02,0\n2011-01-03,9876\n", encoding="utf-8"
    )
    X, dates, ids = gallery.load_bikes(path)
    assert_array_equal(X[:, 0], [1234, 0, 9876])
    assert_array_equal(ids, [0, 1, 2])
    assert_array_equal(dates, ["2011-01-01", "2011-01-02", "2011-01-03"])
    assert_array_equal(gallery.chain_graph(3).toarray(), [[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    path.write_text("dteday,cnt\n2011-01-02,2\n2011-01-01,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="increasing date"):
        gallery.load_bikes(path)


@pytest.mark.parametrize(
    "loss,values",
    [
        ("huber", [0.0, 1.0, 3.0, 6.0]),
        ("logistic", [0.0, 0.0, 1.0, 1.0]),
        ("poisson", [1.0, 2.0, 7.0, 9.0]),
    ],
)
def test_fitted_mean_mapping_and_certification(gallery, loss, values):
    X = np.asarray(values)[:, None]
    result, certificate = gallery.fit_generalized(X, gallery.chain_graph(4), loss, gamma=1.0)
    assert certificate["converged"]
    assert certificate["kkt_residual"] <= 1e-6
    assert certificate["relative_gap"] <= 1e-6
    expected = (
        result.centers
        if loss == "huber"
        else expit(result.centers)
        if loss == "logistic"
        else np.exp(result.centers)
    )
    assert_allclose(result.fitted_means, expected)
    if loss == "logistic":
        assert ((result.fitted_means > 0) & (result.fitted_means < 1)).all()
    if loss == "poisson":
        assert (result.fitted_means > 0).all()
