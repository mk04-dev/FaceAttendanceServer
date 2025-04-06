import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN
import os
from scipy.spatial.distance import cosine
from database import get_db
import redis
from schemas import PersonEmbeddingCreate
import pickle
from consts import DATABASES
from repositories.person_embedding_store import PersonEmbeddingStore

# Khởi tạo mô hình FaceNet
embedder = FaceNet()
detector = MTCNN()
# Đường dẫn đến dataset của bạn

embedding_cache = {}
# Khởi tạo camera
cap = cv2.VideoCapture(0)

async def load_employee_data():
    embedding_cache.clear()  # Xóa cache cũ
    for tenant_cd, url in DATABASES.items():
        db = next(get_db(tenant_cd))  # Lấy session cho database hiện tại
        data = PersonEmbeddingStore.get_all_person_embedding(db)
        for item in data:
            encoding = pickle.loads(item.embedding)
            cache_key = f"{tenant_cd}:{item.party_id}"
            embedding_cache[cache_key] = encoding

load_employee_data()
# Dùng asyncio để chạy song song việc so sánh
async def compare_face_with_redis(key, embed, face_encoding):
    dist = cosine(face_encoding, embed)
    return key, dist

frame_counter = 0
fps_interval = 60  # Phát hiện mỗi 5 khung hình

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_counter += 1
    # Chuyển đổi ảnh sang RGB (MTCNN yêu cầu ảnh ở dạng RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Phát hiện các gương mặt trong ảnh
    faces = detector.detect_faces(frame_rgb)
    print('ahhaha')
    # Duyệt qua các gương mặt đã phát hiện
    for face in faces:
        x, y, w, h = face['box']
        
        # Vẽ khung bao quanh gương mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Cắt phần gương mặt trong ảnh
        face_crop = frame_rgb[y:y+h, x:x+w]
        
        # Trích xuất embeddings của gương mặt
        face_encoding = embedder.embeddings([face_crop])[0]
        
        # So sánh với các embeddings đã lưu trữ trong dataset
        min_dist = float('inf')
        recognized_person = None
        
        if frame_counter % fps_interval == 0:
            for key, embed in embedding_cache.items():
                dist = cosine(face_encoding, embed)
                
                if dist < min_dist:
                    min_dist = dist
                    recognized_person = key
        
        # Vẽ box và tên người nhận diện lên ảnh
        if recognized_person is not None:
            cv2.putText(frame, recognized_person, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
        
    # Hiển thị kết quả
    cv2.imshow("Face Recognition", frame)
    
    # Thoát khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()