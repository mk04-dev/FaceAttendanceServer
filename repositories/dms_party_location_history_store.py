from sqlalchemy.orm import Session
from models import DmsPartyLocationHistory
from schemas import DmsPartyLocationHistoryCreate

class DmsPartyLocationHistoryStore:
    @staticmethod
    def create_location_history(db: Session, history: DmsPartyLocationHistoryCreate):
        new_history = DmsPartyLocationHistory(**history.dict())
        db.add(new_history)
        db.commit()
        db.refresh(new_history)
        return new_history

    @staticmethod
    def get_location_history_by_id(db: Session, local_history_party_id: int):
        return db.query(DmsPartyLocationHistory).filter(DmsPartyLocationHistory.local_history_party_id == local_history_party_id).first()
