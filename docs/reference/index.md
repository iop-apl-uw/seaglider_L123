# Reference

API documentation generated directly from the source code's docstrings and type hints.

| Module | Purpose |
|---|---|
| [`sg_l123`](entry-points.md#sg_l123) | Main entry point — performs L2 and L3 processing on Seaglider data and writes L1/L2/L3 netCDF files. |
| [`sg_l123_plot`](entry-points.md#sg_l123_plot) | Entry point for generating heatmap and position plots from L1/L2/L3 netCDF output. |
| [`sg_l123_utils`](internals.md#sg_l123_utils) | L2/L3 processing utility routines used by `sg_l123`. |
| [`sg_l123_files`](internals.md#sg_l123_files) | Validation and loading of the `var_meta.yml` / `mission_meta.yml` configuration files. |
| [`seaglider_utils`](internals.md#seaglider_utils) | Seaglider-specific utility routines. |
| [`utils`](internals.md#utils) | General-purpose utility routines shared across modules. |

See [Entry Points](entry-points.md) for the two command-line tools, and [Internals & Utilities](internals.md) for the supporting modules they're built on.
