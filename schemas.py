"""HTTP request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Selection(BaseModel):
    start_unix: float
    end_unix: float
    e_channel_min: int
    e_channel_max: int


class ImagingRequest(BaseModel):
    selection: Selection
    l1_json: dict = Field(..., description='In-memory L1 CPD JSON from the BSD UI')
    user_hpc_x: float | None = None
    user_hpc_y: float | None = None
