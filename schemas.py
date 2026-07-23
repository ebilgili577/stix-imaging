"""HTTP request/response schemas."""

from pydantic import BaseModel, Field


class Selection(BaseModel):
    start_unix: float
    end_unix: float
    e_channel_min: int
    e_channel_max: int


class UserHpc(BaseModel):
    """Optional user-supplied source location in helioprojective arcsec."""

    hpc_x: float
    hpc_y: float


class ImagingRequest(BaseModel):
    selection: Selection
    l1_json: dict = Field(..., description='In-memory L1 CPD JSON from the BSD UI')
    user_hpc: UserHpc | None = Field(
        default=None,
        description='If set, use this HPC location as the imaging phase center',
    )
