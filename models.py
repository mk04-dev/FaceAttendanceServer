from sqlalchemy import (
    BLOB, Column, String, Integer, DateTime, ForeignKey
)
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

class PersonEmbedding(Base):
    __tablename__ = "person_embedding"
    idx = Column(Integer, primary_key=True, index=True, autoincrement=True)
    party_id = Column(String(20), ForeignKey("party.party_id"), index=True)
    embedding = Column(BLOB, nullable=True)
    created_date = Column(DateTime, default=datetime.utcnow)
    updated_date = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    party = relationship("Party", foreign_keys=[party_id])
    
class Person(Base):
    __tablename__ = "person"

    party_id = Column(String(20), ForeignKey("party.party_id"), primary_key=True, index=True)
    salutation = Column(String(100), nullable=True)
    first_name = Column(String(100), nullable=True, index=True)
    middle_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True, index=True)
    personal_title = Column(String(100), nullable=True)
    suffix = Column(String(100), nullable=True)
    nickname = Column(String(100), nullable=True)
    first_name_local = Column(String(100), nullable=True)
    middle_name_local = Column(String(100), nullable=True)
    last_name_local = Column(String(100), nullable=True)
    other_local = Column(String(100), nullable=True)
    member_id = Column(String(20), nullable=True)
    # birth_date = Column(Date, nullable=True)
    # deceased_date = Column(Date, nullable=True)
    # height = Column(Double, nullable=True)
    # weight = Column(Double, nullable=True)
    # mothers_maiden_name = Column(String(255), nullable=True)
    # social_security_number = Column(String(255), nullable=True)
    # passport_number = Column(String(255), nullable=True)
    # passport_expire_date = Column(Date, nullable=True)
    # total_years_work_experience = Column(Double, nullable=True)
    # comments = Column(String(255), nullable=True)
    # employment_status_enum_id = Column(String(20), ForeignKey("enumeration.enum_id"), nullable=True)
    # residence_status_enum_id = Column(String(20), ForeignKey("enumeration.enum_id"), nullable=True)
    # occupation = Column(String(100), nullable=True)
    # years_with_employer = Column(DECIMAL(20, 0), nullable=True)
    # months_with_employer = Column(DECIMAL(20, 0), nullable=True)
    # existing_customer = Column(String(1), nullable=True)
    # card_id = Column(String(60), unique=True, nullable=True)
    # created_by = Column(String(20), ForeignKey("party.party_id"), nullable=True)
    # updated_by = Column(String(20), ForeignKey("party.party_id"), nullable=True)
    # created_date = Column(DateTime, default=datetime.utcnow)
    # updated_date = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # last_updated_stamp = Column(DateTime, nullable=True)
    # last_updated_tx_stamp = Column(DateTime, nullable=True)
    # created_stamp = Column(DateTime, nullable=True)
    # created_tx_stamp = Column(DateTime, nullable=True)
    # sex = Column(String(10), nullable=True)
    # nationality = Column(String(50), nullable=True)
    # ethnicity = Column(String(50), nullable=True)
    # religion = Column(String(50), nullable=True)
    # marital_status = Column(String(10), nullable=True)
    # subject_of_labour = Column(String(10), nullable=True)
    # bank_enabled = Column(Boolean, nullable=True)
    # dependent_person_number = Column(Integer, nullable=True, default=0)
    # insurance_code = Column(String(50), nullable=True)
    # tax_code = Column(String(50), nullable=True)
    # first_date = Column(Date, nullable=True)
    # last_date = Column(Date, nullable=True)
    # s3_id = Column(Integer, ForeignKey("s3_file.id"), nullable=True)
    # address = Column(String(255), nullable=True)
    # special_rate = Column(Boolean, nullable=True, default=False)
    # social_insurance_workers = Column(DECIMAL(6, 2), nullable=True)
    # health_insurance_workers = Column(DECIMAL(6, 2), nullable=True)
    # accident_insurance_workers = Column(DECIMAL(6, 2), nullable=True)
    # union_funds_workers = Column(DECIMAL(6, 2), nullable=True)
    # social_insurance_employer = Column(DECIMAL(6, 2), nullable=True)
    # health_insurance_employer = Column(DECIMAL(6, 2), nullable=True)
    # accident_insurance_employer = Column(DECIMAL(6, 2), nullable=True)
    # union_funds_employer = Column(DECIMAL(6, 2), nullable=True)
    # hanet_timekeeper = Column(Boolean, nullable=True, default=False)
    # hanet_place_id = Column(Integer, nullable=True)
    # permanent_address = Column(String(255), nullable=True)
    # timekeeping_id = Column(String(100), nullable=True)
    # academic_title = Column(String(100), nullable=True)
    # degree = Column(String(100), nullable=True)

    # # Quan hệ với các bảng khác
    party = relationship("Party", foreign_keys=[party_id])
    # created_by_user = relationship("Party", foreign_keys=[created_by])
    # updated_by_user = relationship("Party", foreign_keys=[updated_by])
    # employment_status = relationship("Enumeration", foreign_keys=[employment_status_enum_id])
    # residence_status = relationship("Enumeration", foreign_keys=[residence_status_enum_id])
    # s3_file = relationship("S3File", foreign_keys=[s3_id])

class DmsPartyLocationHistory(Base):
    __tablename__ = "dms_party_location_history"

    local_history_party_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    party_id = Column(String(20), ForeignKey("party.party_id"), nullable=True)
    geo_point_id = Column(String(20), nullable=True)
    content_id = Column(String(20), nullable=True)
    # geo_point_id = Column(String(20), ForeignKey("geo_point.geo_point_id"), nullable=True)
    # content_id = Column(String(20), ForeignKey("content.content_id"), nullable=True)
    # product_store_id = Column(String(20), ForeignKey("product_store.product_store_id"), nullable=True)
    # app_installation_id = Column(Integer, ForeignKey("app_installation.id"), nullable=True)
    # created_by = Column(String(20), ForeignKey("party.party_id"), nullable=True)
    # updated_by = Column(String(20), ForeignKey("party.party_id"), nullable=True)
    created_date = Column(DateTime, default=datetime.utcnow)
    updated_date = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # recorded_by_party_id = Column(String(20), ForeignKey("party.party_id"), nullable=True)
    last_updated_stamp = Column(DateTime, nullable=True)
    last_updated_tx_stamp = Column(DateTime, nullable=True)
    created_stamp = Column(DateTime, nullable=True)
    created_tx_stamp = Column(DateTime, nullable=True)
    note = Column(String(255), nullable=True)
    source_timekeeping = Column(String(255), nullable=True)
    # register_business_trip_id = Column(Integer, ForeignKey("register_business_trip.id"), nullable=True)
    branch_id = Column(String(20), nullable=True)

    # Quan hệ với các bảng khác
    party = relationship("Party", foreign_keys=[party_id])
    # geo_point = relationship("GeoPoint", foreign_keys=[geo_point_id])
    # content = relationship("Content", foreign_keys=[content_id])
    # product_store = relationship("ProductStore", foreign_keys=[product_store_id])
    # app_installation = relationship("AppInstallation", foreign_keys=[app_installation_id])
    # created_by_user = relationship("Party", foreign_keys=[created_by])
    # updated_by_user = relationship("Party", foreign_keys=[updated_by])
    # recorded_by_party = relationship("Party", foreign_keys=[recorded_by_party_id])
    # register_business_trip = relationship("RegisterBusinessTrip", foreign_keys=[register_business_trip_id])

class Party(Base):
    __tablename__ = "party"

    party_id = Column(String(20), primary_key=True)
    party_type_id = Column(String(20))
    # party_type_id = Column(String(20), ForeignKey("party_type.party_type_id"))
    external_id = Column(String(20))
    global_id = Column(String(250))
    # TOTP_SECRET_KEY = Column(String(250))
    # PREFERRED_CURRENCY_UOM_ID = Column(String(20), ForeignKey("uom.UOM_ID"))
    # DESCRIPTION = Column(String)  
    # STATUS_ID = Column(String(20), ForeignKey("status_item.STATUS_ID"))
    # CREATED_BY = Column(String(20), ForeignKey("party.party_id"))
    # UPDATED_BY = Column(String(20), ForeignKey("party.party_id"))
    # UPDATED_DATE = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # CREATED_DATE = Column(DateTime, default=datetime.utcnow)
    # CREATED_BY_USER_LOGIN = Column(String(250), ForeignKey("user_login.USER_LOGIN_ID"))
    # LAST_MODIFIED_DATE = Column(DateTime)
    # LAST_MODIFIED_BY_USER_LOGIN = Column(String(250), ForeignKey("user_login.USER_LOGIN_ID"))
    # DATA_SOURCE_ID = Column(String(20), ForeignKey("data_source.DATA_SOURCE_ID"))
    # IS_UNREAD = Column(CHAR(1))
    # IS_REWARDS_CUSTOMER = Column(CHAR(1))
    # IS_EXTERNAL_PARTY = Column(CHAR(1))