# -*- python-fmt -*-
## Copyright (c) 2023, 2024  University of Washington.
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:
##
## 1. Redistributions of source code must retain the above copyright notice, this
##    list of conditions and the following disclaimer.
##
## 2. Redistributions in binary form must reproduce the above copyright notice,
##    this list of conditions and the following disclaimer in the documentation
##    and/or other materials provided with the distribution.
##
## 3. Neither the name of the University of Washington nor the names of its
##    contributors may be used to endorse or promote products derived from this
##    software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE UNIVERSITY OF WASHINGTON AND CONTRIBUTORS “AS
## IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
## DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF WASHINGTON OR CONTRIBUTORS BE
## LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
## CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
## GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
## HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
## LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
## OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Routines related to input and validation of config and metadata files
"""

import enum
from dataclasses import field

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
)
from pydantic.dataclasses import dataclass

from utils import AttributeDict


@dataclass(config=dict(extra="forbid"))
class ProcessingConfig:
    # Days
    despike_running_mean_dx: float = field(default=3.0)
    # Meters
    despike_running_mean_dy: float = field(default=10.0)
    # Percentage of whole data range to calculate statistics over
    data_range: float = field(default=0.95)
    # Removes/interpolates points greater then this from the mean (std dev)
    despike_deviations_for_mean: float = field(default=2.0)
    # max gap to interpolate over is less then this (meters)
    max_depth_gap: float = field(default=50.0)
    # True - do not include dives that are missing; False they are left as NaN profiles
    remove_missing_dives: bool = field(default=False)
    # Hack for SG219 (and maybe others) to mark the downcast a bad for the ocr504i
    ocr504i_hack: bool = field(default=False)


class GlobalAttributes(BaseModel):
    time_coverage_resolution: str = field(default="PT1S")
    # TODO - add in the remaining required fields

    # Allow anything not in the model
    model_config = ConfigDict(extra="allow")


class MissionModel(BaseModel):
    processing_config: ProcessingConfig | None
    global_attributes: GlobalAttributes | None


def load_mission_meta(mission_meta_filename, logger):
    """loads and validates mission meta filename"""

    with open(mission_meta_filename, "r") as fi:
        mission_dict = yaml.safe_load(fi)

    for k in ("processing_config", "global_attributes"):
        if k not in mission_dict:
            mission_dict[k] = {}

    try:
        mission_model = MissionModel(**mission_dict)
    except ValidationError as e:
        for error in e.errors():
            location = f"{':'.join([x for x in error['loc']])}"
            logger.error(f"In {mission_meta_filename} - {location}, {error['msg']}, input:{error['input']}")
        return (None, None)

    # Note: Since mission_model.processing_config is a dataclass, not a pydantic model,
    # it cannot be wrapped with an AttributeDict since is it not iterable.  This isn't an issue
    # for the use in the main code.
    return (mission_model.processing_config, AttributeDict(mission_model.global_attributes))


class NCDataType(enum.Enum):
    f: str = "f"
    d: str = "d"
    i: str = "i"
    s: str = "s"
    b: str = "b"


# Models for var_meta.yml file contents
class NCCoverageContentType(enum.Enum):
    physicalMeasurement: str = "physicalMeasurement"
    coordinate: str = "coordinate"
    modelResult: str = "modelResult"
    auxiliaryInformation: str = "auxiliaryInformation"
    qualityInformation: str = "qualityInformation"
    referenceInformation: str = "referenceInformation"


class NCAttribs(BaseModel):
    FillValue: float | None = None
    description: StrictStr | None = None
    l1: StrictStr | None = None
    l2: StrictStr | None = None
    l3: StrictStr | None = None
    axis: StrictStr | None = None
    coordinates: StrictStr | None = None
    instrument: StrictStr | None = None
    units: StrictStr | None
    coverage_content_type: NCCoverageContentType
    comments: StrictStr | None = None
    standard_name: StrictStr | None = None
    long_name: StrictStr | None
    valid_min: StrictFloat | None = None
    valid_max: StrictFloat | None = None


class NCVarMeta(BaseModel):
    qc_name: StrictStr | None = None
    time_name: StrictStr | None = None
    truck_time_name: StrictStr | None = None
    depth_name: StrictStr | None = None
    despike: bool
    nc_varname: StrictStr
    nc_dimensions: list[StrictStr] = None
    nc_L1_dimensions: list[StrictStr] = None
    nc_attribs: NCAttribs
    nc_type: NCDataType
    decimal_pts: NonNegativeInt | None = None
    include_in_L23: bool | None = True
    include_in_L3: bool | None = True
    model_config = ConfigDict(extra="forbid")


class NCVarAttribs(BaseModel):
    FillValue: float | None = None
    make_model: StrictStr
    coverage_content_type: NCCoverageContentType
    model_config = ConfigDict(extra="forbid")


class NCVar(BaseModel):
    data: StrictInt
    nc_varname: StrictStr
    nc_dimensions: list[StrictStr] = None
    nc_L1_dimensions: list[StrictStr] = None
    nc_attribs: NCVarAttribs
    nc_type: NCDataType
    decimal_pts: NonNegativeInt | None = None
    model_config = ConfigDict(extra="forbid")


def load_instrument_metadata(var_meta_filename, instrument_meta_filename, logger):
    L2_L3_var_meta = {}
    additional_variables = {}
    for filename in (var_meta_filename, instrument_meta_filename):
        if filename is None:
            continue
        try:
            with open(filename, "r") as fi:
                orig = yaml.safe_load(fi)
            for section, accum, model in (
                ("varmeta", L2_L3_var_meta, NCVarMeta),
                ("variables", additional_variables, NCVar),
            ):
                if section in orig:
                    for k, v in orig[section].items():
                        try:
                            if isinstance(v, dict):
                                m = model(**v)
                            else:
                                logger.error(f"{k} in {filename} is not a dictionary")
                                continue
                        except ValidationError as e:
                            for error in e.errors():
                                location = f"{k}:{':'.join([x for x in error['loc']])}"
                                logger.error(f"In {filename} - {location}, {error['msg']}, input:{error['input']}")
                                # pdb.set_trace()
                            logger.error(f"Skipping {k}")
                        else:
                            accum[k] = AttributeDict(m.dict())
        except Exception:
            logger.execption(f"Problems processing {filename}")

    return (L2_L3_var_meta, additional_variables)
