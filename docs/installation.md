# Install and build the documentation

The development package requires Python 3.10 or later, NumPy, SciPy and
scikit-learn. There is no published PyPI release yet. Install from a reviewed
checkout of the repository:

```sh
python -m venv .venv
# Activate this environment using your shell's normal activation command.
python -m pip install .
```

Use `python -m pip install -e '.[dev,examples]'` when modifying the library.
CVXPY, pytest and pandas are test dependencies; Matplotlib is an example
extra. R and external reference packages are not runtime dependencies.

## Build this documentation

```sh
python -m pip install '.[docs]'
python -m sphinx -b html -W --keep-going docs build/docs/html
python -m http.server 8000 --directory build/docs/html
```

Open `http://localhost:8000`. The build treats warnings as errors and checks
internal document references. The API pages import the installed package;
the configuration does not add the checkout's `src` to Python's search path.
Use a clean, matching package installation when checking release artifacts.

The HTML contains search, rendered scientific tables, downloadable data and
source links. Mathematical notation uses MathJax, loaded from its configured
public CDN when a browser displays an equation. Text and code remain available
without that script; a fully offline math bundle is not included.

The [API guide](api.md) explains conventions and supported combinations.
The [generated API reference](api_generated.rst) reflects current public
signatures and NumPy-style docstrings. Both are checked during documentation
builds so changes to code and guidance remain reviewable together.

## Validation and release

From the checkout, install the test dependencies with
`python -m pip install '.[test]'`, then run `python -m pytest -q` for numerical
and API tests. The development extra already includes these dependencies;
the base and documentation extras do not. See the
[distribution validation](distribution_validation.md) for testing built
artifacts from outside the checkout, and the [release process](releasing.md)
for candidate validation and publishing steps. Building this HTML or a wheel
does not publish either artifact.
