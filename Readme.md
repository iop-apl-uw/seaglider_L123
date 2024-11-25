# Seaglider Data L123 Processing Overview

Software for processing 
[Seaglider](https://iop.apl.washington.edu/seaglider.php) output data into level one, two and three 
data products.  For a description of the levels and processing see the document See [Seaglider_data_L123.pdf](Seaglider_data_L123.pdf?raw=true) in the docs directory for further details.
This software is developed at the University of Washington,
maintained and supported by the [IOP group at APL-UW](https://iop.apl.washington.edu/index.php).

#  Setup

Clone from github or copy everything in this directory tree locally.  You are strongly advised to setup 
a python virtual environment to run this code.

To install dependencies:

```pip install -r requirements.txt```

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

```python seaglider_L123.py [-h] [--verbose] --profile_dir PROFILE_DIR --L123_dir L123_DIR --base_name BASE_NAME [--var_meta VAR_META] --mission_meta MISSION_META```

where

- ```--profile_dir``` is the location on disk of the seaglider per-dive netCDF files
- ```--L123_dir``` is the location to write the output L1, L2 and L3 netCDF files
- ```--base_name``` is the name of the mission - ```sg249_NANOOS_Apr24``` for examples
- ```--mission_meta``` is the path to you copy of the ```mission_meta.yml``` file
- ```--var_meta``` is the optional path to an updated version of ```var_meta.yml``` (if extra instruments have been added, or the core metadata adjusted)

