# Building the Docs

This site is built with [MkDocs](https://www.mkdocs.org/) using the
[Material](https://squidfunk.github.io/mkdocs-material/) theme and
[mkdocstrings](https://mkdocstrings.github.io/) for the [Reference](../reference/index.md)
section.

Install the doc-building dependencies with:

```sh
uv sync --extra dev
```

## Build

```sh
make doc
```

runs `uv run mkdocs build`, producing a static site in `site/`.

## Live preview

```sh
make docs-serve
```

runs `uv run mkdocs serve`, which builds the site and serves it locally with live-reload — pages
update in the browser as you edit files under `docs/` or `mkdocs.yml`.

## Clean

```sh
make docs-clean
```

removes the `site/` build output.

## How the Reference section is generated

The pages under `docs/reference/` contain `::: module_name` directives
(`docs/reference/entry-points.md`, `docs/reference/internals.md`). mkdocstrings resolves each
directive against the corresponding top-level `.py` module and renders its module/class/function
docstrings and type hints directly — there's nothing to keep in sync by hand. Editing a
docstring in the source code is picked up automatically the next time the docs are built or
served.
