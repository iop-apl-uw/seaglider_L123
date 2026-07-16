# Configuration

The `config` directory holds two configuration files used to drive processing.

## `var_meta.yml`

The needed variable definitions and metadata used to process a typical Seaglider dataset and
generate the level one, two and three netCDF files. Currently supported instruments are CTD,
Seabird and Aanderaa optode, and Wetlabs backscatter (not every variant is currently covered). At
present, adding new instruments requires duplicating this file and adding the needed entries —
documentation on how to do so is coming soon.

## `mission_meta.yml`

An example file, intended to be duplicated and tailored to the particular mission being
processed. It has two top-level sections:

- `processing_config` — parameters controlling the processing itself (see table below). The
  default values are generally good for most Seaglider data, but can be adjusted to fit your
  needs.
- `global_attributes` — copied directly into the global attributes of the output netCDF files.
  See the attributes preceded by `#Update` for attributes that need editing for your dataset.

### `processing_config` parameters

Defined by the `ProcessingConfig` dataclass in [`sg_l123_files`](../reference/internals.md#sg_l123_files).

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `despike_running_mean_dx` | float | `3.0` | Running-mean window width, in days. |
| `despike_running_mean_dy` | float | `10.0` | Running-mean window height, in meters. |
| `data_range` | float | `0.95` | Percentage of the whole data range used to calculate statistics over. |
| `despike_deviations_for_mean` | float | `2.0` | Points more than this many standard deviations from the mean are removed/interpolated. |
| `max_depth_gap` | float | `50.0` | Maximum gap, in meters, that will be interpolated over. |
| `remove_missing_dives` | bool | `False` | If `True`, dives that are missing are dropped entirely rather than left in as NaN profiles. |
| `ocr504i_hack` | bool | `False` | Hack for SG219 (and possibly others) to mark the downcast as bad for the OCR504i instrument. |

For more background on the processing levels and the effect of these parameters, see
[Seaglider_data_L123.pdf](../Seaglider_data_L123.pdf).
