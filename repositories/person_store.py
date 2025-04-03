from sqlalchemy.orm import Session
from models import Person
from schemas import PersonCreate

class PersonStore:
    @staticmethod
    def create_person(db: Session, person: PersonCreate):
        new_person = Person(**person.dict())
        db.add(new_person)
        db.commit()
        db.refresh(new_person)
        return new_person

    @staticmethod
    def get_person_by_id(db: Session, party_id: str):
        return db.query(Person).filter(Person.party_id == party_id).first()
