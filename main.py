from typing import List
from fastapi import FastAPI, File, UploadFile, Form, Request
import uvicorn
import redis
import cv2
import numpy as np
from database import SessionLocal
from schemas import PersonEmbeddingCreate
from repositories.person_embedding_store import PersonEmbeddingStore
from keras_facenet import FaceNet
import pickle
from scipy.spatial.distance import cosine
import asyncio
import logging

app = FastAPI()

# Kết nối Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

embedder = FaceNet()  # Load model FaceNet

# Kết nối MySQL
db = SessionLocal()
LOG = logging.getLogger(__name__)

# Hàm tải dữ liệu nhân viên từ DB vào cache
async def load_employee_data():
    redis_client.flushdb()  # Xóa cache cũ
    data = PersonEmbeddingStore.get_all_person_embedding(db)
    for item in data:
        encoding = pickle.loads(item.embedding)
        redis_client.set(item.party_id, pickle.dumps(encoding))

@app.get("/reload-cache")
async def reload_cache():
    await load_employee_data()
    return {"message": "Reloaded face data"}

@app.on_event("startup")
async def startup_event():
    await load_employee_data()

@app.post("/recognize")
async def recognize_face(image: UploadFile = File(...)):
    LOG.info("recognize_face")
    """API nhận diện khuôn mặt"""
    image_bytes = await image.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # Chuyển ảnh sang RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Trích xuất đặc trưng khuôn mặt
    face_encoding = embedder.embeddings([rgb_image])[0]

    # # So sánh với dữ liệu trong Redis
    # min_dist = 1.0
    # best_match = "Unknown"

    # for key in redis_client.keys():
    #     known_encoding = pickle.loads(redis_client.get(key))
    #     dist = cosine(face_encoding, known_encoding)
    #     if dist < min_dist:
    #         min_dist = dist
    #         best_match = key.decode()

    # Lấy danh sách tất cả các key từ Redis một cách bất đồng bộ
    redis_keys = list(redis_client.keys())

    # Dùng asyncio để chạy song song việc so sánh
    async def compare_face_with_redis(key):
        known_encoding = pickle.loads(redis_client.get(key))
        dist = cosine(face_encoding, known_encoding)
        return key.decode(), dist

    # Chạy song song các tác vụ nhận diện
    comparisons = await asyncio.gather(*[compare_face_with_redis(key) for key in redis_keys])

    # Tìm kết quả tốt nhất
    best_match, min_dist = min(comparisons, key=lambda x: x[1])

    return {"name": best_match, "score": min_dist}

@app.post("/add_employee")
async def add_employee(request: Request):
    """
    Endpoint để thêm nhân viên mới.
    Client gửi các file hình với các key "image0", "image1", ... và dữ liệu form "name".
    Server sẽ:
      1. Lấy tên nhân viên từ form.
      2. Duyệt qua các key bắt đầu bằng "image" để đọc nội dung file.
      3. Decode ảnh, chuyển sang RGB và trích xuất encoding từ FaceNet.
      4. Tính trung bình encoding từ các ảnh và lưu vào MySQL cũng như cập nhật cache Redis.
    """
    form = await request.form()  # Lấy toàn bộ form data
    party_id = form.get("party_id")
    encodings = []

     # Duyệt qua các key của form data
    for key in form.keys():
        if key.startswith("image"):
            file = form[key]  # file là UploadFile
            image_bytes = await file.read()
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if img is None:
                continue
            # Chuyển ảnh sang RGB (FaceNet thường yêu cầu ảnh RGB)
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Trích xuất encoding, giả sử mỗi ảnh có đúng 1 khuôn mặt
            encoding = embedder.embeddings([rgb_image])[0]
            encodings.append(encoding)
    
    if not encodings:
        return {"error": "Không nhận được hình hợp lệ để trích xuất khuôn mặt."}
    
    # Tính trung bình các encoding để có vector đại diện cho nhân viên
    avg_encoding = np.mean(encodings, axis=0)
    encoding_blob = pickle.dumps(avg_encoding)

    
    # Lưu vào database
    try:
        entity = PersonEmbeddingCreate(party_id=party_id, embedding=encoding_blob)
        exist = PersonEmbeddingStore.get_person_embedding_by_id(db, party_id)

        if exist:
            PersonEmbeddingStore.update_person_embedding(db, party_id, entity)
        else:
            PersonEmbeddingStore.create_person_embedding(db, entity)

    except Exception as e:
        return {"error": f"Lỗi khi lưu vào MySQL: {e}"}
    
    # Cập nhật cache Redis ngay lập tức
    try:
        redis_client.set(party_id, encoding_blob)
    except Exception as e:
        return {"error": f"Lỗi khi cập nhật Redis: {e}"}

    return {"message": f"Added employee {party_id}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="trace")
