# Code Style & Standards

This project targets **Python 3.13**.

## Python standards

- **Type hinting** is mandatory for all definitions. Use the standard built-in collections
  natively (`list`, `dict`, `set`, `tuple`) and `|` for unions, rather than `typing.List` /
  `typing.Optional` etc.
- **Modern generics**: use PEP 695 syntax when creating generics (`def function[T](arg: T) -> T:`)
  rather than the older `TypeVar(...)` binding style.
- **Docstrings** conform to the Google Python Style Guide. Every public function includes `Args:`,
  `Returns:`, and `Raises:` clauses, as rendered on the [Reference](../reference/index.md) pages.
- **Bypassing type errors**: if a type error can't be cleanly refactored due to a dynamic
  implementation detail, use a targeted `# ty: ignore[rule-name]` comment rather than a generic
  `# type: ignore`.

## Filesystem paths

Use `pathlib.Path` for all filesystem paths — never `os.path`, string concatenation, or
`os.system`/`shutil` calls with raw path strings.

- Joining: `Path(base) / "subdir" / "file.txt"`, not `os.path.join(...)`
- Existence/type checks: `.exists()`, `.is_file()`, `.is_dir()`
- Reading/writing: `.read_text()`, `.write_text()`, `.read_bytes()` instead of manual `open()` where feasible
- Globbing: `.glob()` / `.rglob()` instead of `os.walk` or `glob.glob`
- When a third-party API strictly requires a `str`, cast explicitly at the boundary (`str(path)`),
  with a short comment noting why.

## Validation commands

| Command | Makefile shortcut | Purpose |
|---|---|---|
| `uv run ruff check --select I --fix *py tests/*py` + `uv run ruff format *py tests/*py` | `make rufffmt` | Import sorting and formatting. |
| `uv run ruff check .` | `make rufflint` | Linting. |
| `uv run ty check --output-format github` | `make typecheck` | Static type checking. |

`make all` runs formatting, linting, type checking, and the test suite in sequence.
