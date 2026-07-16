# Running L2/L3 Processing

`sg_l123.py` performs L2 and L3 processing on Seaglider data and writes L1/L2/L3 netCDF files:

```sh
uv run sg_l123.py [-h] [--verbose] --profile_dir PROFILE_DIR --L123_dir L123_DIR \
    --base_name BASE_NAME [--var_meta VAR_META] --mission_meta MISSION_META \
    [--instrument_meta INSTRUMENT_META] \
    [--debug_pdb | --no-debug_pdb] \
    [--skip_processing_errors | --no-skip_processing_errors] \
    [--do_plots | --no-do_plots] \
    [--do_plots_detailed | --no-do_plots_detailed] \
    [--interactive | --no-interactive]
```

where

| Flag | Meaning |
|---|---|
| `--profile_dir` | Location on disk of the Seaglider per-dive netCDF files. |
| `--L123_dir` | Location to write the output L1, L2 and L3 netCDF files. |
| `--base_name` | Name of the mission — `sg249_NANOOS_Apr24`, for example. |
| `--mission_meta` | Path to your copy of the `mission_meta.yml` file. |
| `--var_meta` | Optional path to an updated version of `var_meta.yml` (if extra instruments have been added, or the core metadata adjusted). |
| `--instrument_meta` | Optional path to instrument-specific metadata, for instruments not covered in `var_meta.yml`. |
| `--debug_pdb` | Drops into the debugger on selected exceptions (default off). |
| `--skip_processing_errors` | Skips per-dive netCDF files flagged as having processing errors (default on). |
| `--do_plots` | Generates diagnostic plots of the L2/L3 processing (default off). |
| `--do_plots_detailed` | Generates more detailed diagnostic plots (default off). |
| `--interactive` | Opens generated plots in a browser as they're created, rather than only writing them to disk (default off). |

See [Configuration](configuration.md) for details on `--mission_meta` and `--var_meta`, and
[Plotting](running-plots.md) for generating heatmap/position plots once processing has finished.
