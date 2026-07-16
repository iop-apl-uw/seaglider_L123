# Testing & Coverage

Install the testing dependencies with:

```sh
uv sync --extra ci
```

Run the test suite with coverage:

```sh
uv run pytest --cov --cov-report=term-missing tests/
```

or, using the Makefile shortcut:

```sh
make test
```

`make testhtml` runs the same suite but writes an HTML coverage report instead of printing missing
lines to the terminal — useful for browsing coverage gaps interactively.

## Coverage targets

Minimum test coverage is enforced at **85%** project-wide (see the `[tool.coverage.report]`
section of `pyproject.toml`). Core logic / the service layer is held to a stricter 100% target.
