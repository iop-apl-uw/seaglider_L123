# Development

This section covers setting up a development environment, running the test suite, the project's
coding standards, and building this documentation site.

## Dependency groups

`pyproject.toml` defines two optional dependency groups, named by what they're used for rather
than by "prod vs dev":

- `ci` — the test/lint/typecheck toolchain (pytest, pytest-cov, ruff, ty).
- `dev` — the documentation toolchain (mkdocs, mkdocs-material, mkdocstrings, markdown-include).

To get everything needed for full-project development, including building these docs:

```sh
uv sync --extra ci --extra dev
```

See [Testing & Coverage](testing.md), [Code Style & Standards](code-style.md), and
[Building the Docs](building-docs.md) for details on each toolchain.
