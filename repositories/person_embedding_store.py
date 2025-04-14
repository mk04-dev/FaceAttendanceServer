from sqlalchemy.orm import Session
from models import PersonEmbedding
from schemas import PersonEmbeddingCreate

class PersonEmbeddingStore:
    @staticmethod
    def create_person_embedding(db: Session, embedding: PersonEmbeddingCreate):
        new_embedding = PersonEmbedding(**embedding.dict())
        db.add(new_embedding)
        db.commit()
        db.refresh(new_embedding)
        return new_embedding

    @staticmethod
    def update_person_embedding(db: Session, party_id: str, embedding: PersonEmbeddingCreate):
        db.query(PersonEmbedding).filter(PersonEmbedding.party_id == party_id).update(embedding.model_dump())
        db.commit()
        return db.query(PersonEmbedding).filter(PersonEmbedding.party_id == party_id).first()

    @staticmethod
    def get_person_embedding_by_id(db: Session, party_id: str):
        return db.query(PersonEmbedding).filter(PersonEmbedding.party_id == party_id).first()
    
    @staticmethod
    def get_all_person_embedding(db: Session):
        return db.query(PersonEmbedding).all()
    
    @staticmethod
    def delete_person_embedding(db: Session, party_id: str):
        db.query(PersonEmbedding).filter(PersonEmbedding.party_id == party_id).delete()
        db.commit()