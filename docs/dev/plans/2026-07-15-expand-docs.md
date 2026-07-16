# Expand project documentation (`make doc`)

## Context

The project currently builds a minimal MkDocs site (`make doc` → `uv run mkdocs build`) consisting of exactly two content files: `docs/index.md` (a raw include of `Readme.md`) and `docs/reference.md` (mkdocstrings autodoc for 5 of the 6 top-level modules — `utils.py` is missing). There's no dedicated user guide, no development/contributing section in the site, and no navigation structure beyond two flat entries.

The goal is to expand this into three real sections — **User Guide**, **Reference**, and **Development** — as requested, while keeping `Readme.md` itself untouched (per user decision) and without transcribing the `Seaglider_data_L123.pdf` into web pages (it stays a linked download, per user decision). The docs site remains local-build-only for now (no GitHub Pages / deploy target).

## Decisions from scoping

- **Readme.md stays as-is** — not trimmed. User Guide pages restructure/adapt its content into the docs site, but the root README file is not edited.
- **PDF stays a download-only artifact** — no `background.md`/`data-products.md` transcription pages. Docs pages that reference it just link to it (as the README already does).
- **Local-only docs** — no `site_url`/`repo_url`/`gh-deploy` wiring; just `make doc` (build) and a new `make docs-serve` (live preview).
- **Branch**: `chore/expand-docs`, created by the user.
- Verified `sg_l123_files.py`'s `ProcessingConfig` dataclass (lines 51-68) as the authoritative source for the config parameter table: `despike_running_mean_dx` (days, default 3.0), `despike_running_mean_dy` (meters, default 10.0), `data_range` (fraction, default 0.95), `despike_deviations_for_mean` (default 2.0), `max_depth_gap` (meters, default 50.0), `remove_missing_dives` (bool, default False), `ocr504i_hack` (bool, default False) — each with an inline comment explaining it.

## Target structure

```
docs/
├── index.md                          # new, short landing page (not a raw README include)
├── user-guide/
│   ├── index.md                      # section overview
│   ├── installation.md               # from README "Setup"
│   ├── configuration.md              # from README "Configuration files" + ProcessingConfig table + link to PDF
│   ├── running-processing.md         # from README "Running" (sg_l123.py CLI)
│   └── running-plots.md              # from README "Plotting" (sg_l123_plot.py CLI)
├── reference/
│   ├── index.md                      # short intro + module map
│   ├── entry-points.md               # ::: sg_l123, ::: sg_l123_plot
│   └── internals.md                  # ::: sg_l123_utils, ::: sg_l123_files, ::: seaglider_utils, ::: utils  (utils.py gap fixed)
└── development/
    ├── index.md                      # dev env setup (clarify `--extra ci` vs `--extra dev`)
    ├── testing.md                    # pytest/coverage, make test/testhtml
    ├── code-style.md                 # ruff/ty + CLAUDE.md conventions (type hints, docstrings, pathlib)
    ├── building-docs.md              # make doc / make docs-serve, how mkdocstrings pulls from docstrings
    └── contributing.md               # includes root CONTRIBUTING.md via markdown_include (only remaining include use)
```

`docs/reference.md` is deleted once its two directives are migrated into `reference/entry-points.md` and `reference/internals.md`.

Note on `pyproject.toml` extras (worth calling out explicitly in `development/index.md` since it's non-obvious): `dev` = doc-building toolchain (mkdocs/mkdocstrings/mkdocs-material/markdown-include), `ci` = test/lint/typecheck toolchain (pytest/ruff/ty). Contributors need `uv sync --extra ci --extra dev` to get everything.

## `mkdocs.yml` changes

- `nav:` restructured into the three sections above (Home / User Guide / Reference / Development).
- `theme.features`: `navigation.tabs`, `navigation.sections`, `navigation.top`, `search.suggest`, `content.code.copy` — maps the three sections onto tabs, adds copy buttons for the many CLI code blocks, cheap search/UX wins.
- `theme.palette`: light/dark toggle (standard material recipe, no custom branding).
- `plugins`: add explicit `search` (built into mkdocs core, no new dependency) alongside `mkdocstrings`; add `mkdocstrings.handlers.python.options`: `show_source: true`, `show_root_heading: true`, `docstring_style: google`, `merge_init_into_class: true`, `show_signature_annotations: true` — makes the Google-docstring/type-hint conventions visible on the rendered pages.
- `markdown_extensions`: keep `markdown_include.include` (scoped to `development/contributing.md` only going forward), add `admonition`, `pymdownx.details`, `pymdownx.superfences`, `tables`, `toc: {permalink: true}` — all ship as transitive deps of `mkdocs-material`/mkdocs core already, no `pyproject.toml` change needed.
- No mermaid/PDF-rendering plugins added (nothing in scope needs them).

## Makefile changes

Add two targets, consistent with existing style (`-` prefix, `uv run`):
```make
docs-serve:
	-uv run mkdocs serve

docs-clean:
	-rm -rf site/
```
Keep the existing `doc` target name unchanged (no rename, avoids breaking any existing usage). Confirm `site` stays covered by `.gitignore` (it already is).

## Sequencing

1. ~~User creates/checks out `chore/expand-docs`~~ — done.
2. ~~Save this plan to `docs/dev/plans/`~~ — done (renamed to `2026-07-15-expand-docs.md` to match the updated CLAUDE.md naming convention).
3. ~~Scaffold `docs/user-guide/`, `docs/reference/`, `docs/development/` and update `mkdocs.yml` nav/theme/plugins~~ — done.
4. ~~Migrate reference content: `reference/entry-points.md`, `reference/internals.md` (adding `::: utils`), delete old `docs/reference.md`~~ — done.
5. ~~Write User Guide pages~~ — done.
6. ~~Write Development pages~~ — done.
7. ~~Write `docs/index.md` landing page~~ — done.
8. ~~Add Makefile `docs-serve`/`docs-clean` targets~~ — done.
9. ~~Verify: `make doc`, `make docs-serve` (all nav tabs return 200), `make docs-clean`~~ — done. Also added `exclude_docs: dev/` to `mkdocs.yml` so this plans directory itself isn't published as a site page (mkdocs flagged it as an orphan page not in nav).
10. Remove this task from `PLANS.md` if/when one exists (none currently in the repo) — n/a, no `PLANS.md` exists yet.

**Status: complete.** All verification steps passed: `make doc` builds cleanly, all 6 modules render on the reference pages with expected function/class counts, `make docs-serve` served all nav tabs (200 OK), `make docs-clean` removes `site/`, and `ruff check` / `ruff format --check` / `ty check` all pass with no findings (docs-only change).

## Verification

- `make doc` completes without warnings/errors (nav pointing at missing files or broken mkdocstrings targets both surface here).
- `make docs-serve`, then manually browse every nav entry in a browser, confirming: all 6 modules render on the two reference pages (spot-check against known counts — `sg_l123.py` 6 funcs/2 classes, `sg_l123_plot.py` 3 funcs, `sg_l123_utils.py` 5 funcs, `sg_l123_files.py` 2 funcs/9 classes, `seaglider_utils.py` 6 funcs, `utils.py` 3 funcs/2 classes), the PDF link resolves, and internal cross-links between pages work.
- `uv run ruff check --fix` / `uv run ruff format` / `uv run ty check` still pass (no source `.py` files should need changes, but docstrings aren't touched by this task so this is just a safety check).
