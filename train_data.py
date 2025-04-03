import numpy as np
import cv2
import os
from keras_facenet import FaceNet
import pickle
from schemas import PersonEmbeddingCreate
from repositories.person_embedding_store import PersonEmbeddingStore
from database import SessionLocal


# Kết nối MySQL
db = SessionLocal()

# Load mô hình FaceNet
embedder = FaceNet()

# Thư mục chứa ảnh nhân viên
DATASET_PATH = "dataset"

for emp_id in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, emp_id)

    if os.path.isdir(person_path):  # Kiểm tra nếu là thư mục
        encodings = []

        for filename in os.listdir(person_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Load ảnh
                image_path = os.path.join(person_path, filename)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Trích xuất đặc trưng khuôn mặt
                encoding = embedder.embeddings([image])[0]
                encodings.append(encoding)

            if encodings:
                # Lấy trung bình vector khuôn mặt của tất cả ảnh
                avg_encoding = np.mean(encodings, axis=0)

                # Lưu vào MySQL
                encoding_blob = pickle.dumps(avg_encoding)
                entity = PersonEmbeddingCreate(party_id=emp_id, embedding=encoding_blob)

                exist = PersonEmbeddingStore.get_person_embedding_by_id(db, emp_id)

                if exist:
                    PersonEmbeddingStore.update_person_embedding(db, emp_id, entity)
                else:
                    PersonEmbeddingStore.create_person_embedding(db, entity)


print("✅ Dữ liệu đã được lưu vào MySQL!")
