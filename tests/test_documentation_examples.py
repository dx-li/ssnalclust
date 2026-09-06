"""Execute the published tutorial and verify the plotted scientific claims."""

import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

ROOT = Path(__file__).resolve().parents[1]


def _gallery_module():
    spec = importlib.util.spec_from_file_location(
        "plot_clustering_paths", ROOT / "examples" / "plot_clustering_paths.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_user_guide_python_examples():
    guide = (ROOT / "docs" / "user_guide.md").read_text()
    examples = re.findall(r"```python\n(.*?)```", guide, flags=re.DOTALL)
    assert examples
    namespace = {"__name__": "__documentation_examples__"}
    for index, code in enumerate(examples):
        exec(compile(code, f"user_guide.md example {index + 1}", "exec"), namespace)


@pytest.fixture(scope="module")
def gallery_data():
    return _gallery_module().compute_gallery()


def test_gallery_recovery_and_certificates(gallery_data):
    X, truth, gammas, results, summaries, snapshot = gallery_data
    assert len(gammas) == len(results) == len(summaries)
    assert all(result.converged for result in results)
    assert max(result.relative_gap for result in results) <= 1e-7
    assert max(result.kkt_residual for result in results) <= 1e-7
    assert_array_equal(results[0].centers, X)
    assert summaries[0]["n_clusters"] == len(X)
    assert summaries[-1]["n_clusters"] == 1
    assert_allclose(results[-1].centers, np.broadcast_to(X.mean(axis=0), X.shape), atol=2e-5)
    labels = summaries[snapshot]["labels"]
    assert_array_equal(labels[:, None] == labels[None, :], truth[:, None] == truth[None, :])


def test_gallery_saves_nonempty_four_panel_figure(gallery_data, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure = _gallery_module().make_figure(gallery_data)
    output = tmp_path / "path.png"
    figure.savefig(output, dpi=80)
    assert len(figure.axes) == 4
    assert all(axis.get_xlabel() and axis.get_ylabel() for axis in figure.axes)
    pixels = plt.imread(output)
    assert pixels.shape[0] >= 600 and pixels.shape[1] >= 800
    assert np.std(pixels[..., :3]) > 0.05
    plt.close(figure)
