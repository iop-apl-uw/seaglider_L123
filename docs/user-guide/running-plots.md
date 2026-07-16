# Plotting

Once L2/L3 processing has completed, `sg_l123_plot.py` generates heatmap and position plots from
the output netCDF files:

```sh
uv run sg_l123_plot.py [-h] [--verbose] --L123_dir L123_DIR --base_name BASE_NAME \
    [--plot_contour] [--plot_webp] \
    [--debug_pdb | --no-debug_pdb] \
    [--interactive | --no-interactive]
```

where

| Flag | Meaning |
|---|---|
| `--L123_dir` | Location of the L1/L2/L3 netCDF files produced by `sg_l123.py`. |
| `--base_name` | Same mission base name used when running `sg_l123.py`. |
| `--plot_contour` | Plots contours instead of heatmaps. |
| `--plot_webp` | Additionally writes a `.webp` image alongside each html plot. |
| `--interactive` | Opens generated plots in a browser as they're created. |

Plots are written to a `plots` subdirectory of `--L123_dir`.
