# Seaglider Level 123 Processing

Software for processing [Seaglider](https://iop.apl.washington.edu/seaglider.php) output data
into level one, two and three data products. This software is developed at the University of
Washington, maintained and supported by the
[IOP group at APL-UW](https://iop.apl.washington.edu/index.php).

For background on Seaglider data and a description of the L1/L2/L3 processing levels, see
[Seaglider_data_L123.pdf](Seaglider_data_L123.pdf).

## Quick links

- **[User Guide](user-guide/index.md)** — install the software, configure it for a mission, and
  run L2/L3 processing and plotting.
- **[Reference](reference/index.md)** — API documentation generated from the source code.
- **[Development](development/index.md)** — set up a dev environment, run tests, and follow the
  project's coding standards.

## Quickstart

```sh
uv sync
uv run sg_l123.py --profile_dir PROFILE_DIR --L123_dir L123_DIR \
    --base_name BASE_NAME --mission_meta MISSION_META
```

See [Installation](user-guide/installation.md) and
[Running L2/L3 Processing](user-guide/running-processing.md) for details.
