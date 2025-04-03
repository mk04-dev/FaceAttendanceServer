from pydantic import BaseModel
from datetime import date, datetime
from typing import Optional

class PersonBase(BaseModel):
    party_id: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    birth_date: Optional[date] = None
    nationality: Optional[str] = None
    marital_status: Optional[str] = None
    address: Optional[str] = None
    occupation: Optional[str] = None
    employment_status_enum_id: Optional[str] = None
    residence_status_enum_id: Optional[str] = None

class PersonCreate(PersonBase):
    pass

class PersonResponse(PersonBase):
    created_date: datetime
    updated_date: datetime

class DmsPartyLocationHistoryBase(BaseModel):
    party_id: Optional[str] = None
    geo_point_id: Optional[str] = None
    content_id: Optional[str] = None
    product_store_id: Optional[str] = None
    app_installation_id: Optional[int] = None
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    recorded_by_party_id: Optional[str] = None
    note: Optional[str] = None
    source_timekeeping: Optional[str] = None
    register_business_trip_id: Optional[int] = None
    branch_id: Optional[str] = None

class DmsPartyLocationHistoryCreate(DmsPartyLocationHistoryBase):
    pass

class DmsPartyLocationHistoryResponse(DmsPartyLocationHistoryBase):
    local_history_party_id: int
    created_date: datetime
    updated_date: datetime

class PersonEmbeddingBase(BaseModel):
    party_id: str
    embedding: bytes

class PersonEmbeddingCreate(PersonEmbeddingBase):
    pass

class PersonEmbeddingResponse(PersonEmbeddingBase):
    created_date: datetime
    updated_date: datetime
