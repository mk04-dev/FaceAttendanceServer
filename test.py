from datetime import datetime
from repositories.dms_party_location_history_store import DmsPartyLocationHistoryStore
from database import get_db
from consts import DATABASES, GEO_POINT_ID, BRANCH_ID
from schemas import PersonEmbeddingCreate, DmsPartyLocationHistoryCreate

def add_dms_history(db, party_id) -> bool:
    current_timestamp = datetime.now()
    entity = DmsPartyLocationHistoryCreate()
    entity.party_id = party_id
    entity.geo_point_id = str(GEO_POINT_ID)
    entity.note = "Camera detection"
    entity.source_timekeeping = "Camera detection"
    entity.branch_id = str(BRANCH_ID)
    entity.created_date = current_timestamp
    entity.updated_date = current_timestamp
    entity.created_stamp =current_timestamp
    entity.created_tx_stamp = current_timestamp
    entity.last_updated_stamp = current_timestamp
    entity.last_updated_tx_stamp = current_timestamp
    try:
        DmsPartyLocationHistoryStore.create_location_history(db, entity)
        return True
    except Exception as e:
        print("An error occurred while adding DMS history:", e)
        return False


if __name__ == "__main__":
    # Example usage
    db = next(get_db("tenant1"))  # Replace with your actual database session retrieval
    party_id = "171"  # Replace with the actual party ID you want to use
    result = add_dms_history(db, party_id)
    print(f"DMS history added: {result}")