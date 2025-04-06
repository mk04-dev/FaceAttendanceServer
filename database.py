from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from consts import DATABASES

Base = declarative_base()

# Tạo engine và session factory cho từng database
engines = {name: create_engine(url) for name, url in DATABASES.items()}
SessionFactories = {
    name: sessionmaker(autocommit=False, autoflush=False, bind=engine)
    for name, engine in engines.items()
}

# Hàm get_db cho phép chọn database
def get_db(db_name: str):
    SessionLocal = SessionFactories.get(db_name)
    if SessionLocal is None:
        raise ValueError(f"Database '{db_name}' is not configured.")
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()