# Seaglider Data L123 Processing Overview

Software for processing 
[Seaglider](https://iop.apl.washington.edu/seaglider.php) output data into level one, two and three 
data products.  For a description of the levels and processing see the document See [Seaglider_data_L123.pdf](Seaglider_data_L123.pdf?raw=true) in the docs directory for further details.
This software is developed at the University of Washington,
maintained and supported by the [IOP group at APL-UW](https://iop.apl.washington.edu/index.php).

#  Setup

Clone from github or copy everything in this directory tree locally.  You are strongly advised to use
[uv](https://github.com/astral-sh/uv) to run this code.

To install dependencies:

```uv sync```

# Configuration files

In the ```config``` directory are two configuration files. 

- ```var_meta.yml``` is the needed
variable definitions and metadata used to process a typical seagliders dataset and generate the level
one, two and three netcdf files.  Currently supported instruments are CTD, Seabird and Aandera optode and
Wetlabs backscatter (not every variant is currently covered).  At present, adding new instruments requires 
duplicating this file and adding the needed entries - documentation on how to do so coming soon.

- ```mission_meta.yml``` is and example file, intended to be duplicated and tailored to the particular
mission being processed.  The top section - ```processing_config``` - contains parameters to the processing.
The default values are generally good for most seaglider data, but can be adjusted to fit your needs.  
A description of these can be found in [Seaglider_data_L123.pdf](Seaglider_data_L123.pdf?raw=true).  The section ```global_attributes``` are copied directly 
into the global attributes in the output netcdf files.  See the attributes preceeded by ```#Update``` for 
attributes to be edited for your dataset.

# Running

```uv run sg_l123.py [-h] [--verbose] --profile_dir PROFILE_DIR --L123_dir L123_DIR --base_name BASE_NAME [--var_meta VAR_META] --mission_meta MISSION_META [--instrument_meta INSTRUMENT_META] [--debug_pdb | --no-debug_pdb] [--skip_processing_errors | --no-skip_processing_errors] [--do_plots | --no-do_plots] [--do_plots_detailed | --no-do_plots_detailed] [--interactive | --no-interactive]```

where

- ```--profile_dir``` is the location on disk of the seaglider per-dive netCDF files
- ```--L123_dir``` is the location to write the output L1, L2 and L3 netCDF files
- ```--base_name``` is the name of the mission - ```sg249_NANOOS_Apr24``` for examples
- ```--mission_meta``` is the path to you copy of the ```mission_meta.yml``` file
- ```--var_meta``` is the optional path to an updated version of ```var_meta.yml``` (if extra instruments have been added, or the core metadata adjusted)
- ```--instrument_meta``` is an optional path to instrument-specific metadata, for instruments not covered in ```var_meta.yml```
- ```--debug_pdb``` drops into the debugger on selected exceptions (default off)
- ```--skip_processing_errors``` skips per-dive netCDF files flagged as having processing errors (default on)
- ```--do_plots``` generates diagnostic plots of the L2/L3 processing (default off)
- ```--do_plots_detailed``` generates more detailed diagnostic plots (default off)
- ```--interactive``` opens generated plots in a browser as they're created, rather than only writing them to disk (default off)

# Plotting

Once L2/L3 processing has completed, ```sg_l123_plot.py``` generates heatmap and position plots from the
output netCDF files:

```uv run sg_l123_plot.py [-h] [--verbose] --L123_dir L123_DIR --base_name BASE_NAME [--plot_contour] [--plot_webp] [--debug_pdb | --no-debug_pdb] [--interactive | --no-interactive]```

where

- ```--L123_dir``` is the location of the L1/L2/L3 netCDF files produced by ```sg_l123.py```
- ```--base_name``` is the same mission base name used when running ```sg_l123.py```
- ```--plot_contour``` plots contours instead of heatmaps
- ```--plot_webp``` additionally writes a ```.webp``` image alongside each html plot
- ```--interactive``` opens generated plots in a browser as they're created

Plots are written to a ```plots``` subdirectory of ```--L123_dir```.

# Development

Install the testing and linting dependencies with:

```uv sync --extra ci```

- Run the test suite with coverage: ```uv run pytest --cov --cov-report=term-missing tests/``` (or ```make test```)
- Lint: ```uv run ruff check --fix``` (or ```make rufflint```)
- Type check: ```uv run ty check``` (or ```make typecheck```)

Minimum test coverage is enforced at 85% project-wide (see ```pyproject.toml```).

