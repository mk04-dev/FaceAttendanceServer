from datetime import datetime
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
import uvicorn
import redis
import cv2
import numpy as np
from database import get_db
from schemas import PersonEmbeddingCreate, DmsPartyLocationHistoryCreate
from repositories.person_embedding_store import PersonEmbeddingStore
from repositories.dms_party_location_history_store import DmsPartyLocationHistoryStore
from keras_facenet import FaceNet
import pickle
from scipy.spatial.distance import cosine
import asyncio
import logging
from consts import DATABASES

class RecorgnizeFace:
    def __init__(self, party_id: str, score: float, added: str):
        self.party_id = party_id
        self.score = score
        self.added = added
    
app = FastAPI()

# Kết nối Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

embedder = FaceNet()  # Load model FaceNet

history = {}
LOG = logging.getLogger(__name__)

# Hàm tải dữ liệu nhân viên từ DB vào cache
async def load_employee_data():
    redis_client.flushdb()  # Xóa cache cũ

    for tenant_cd, url in DATABASES.items():
        db = next(get_db(tenant_cd))  # Lấy session cho database hiện tại
        data = PersonEmbeddingStore.get_all_person_embedding(db)
        for item in data:
            encoding = pickle.loads(item.embedding)
            redis_key = f"{tenant_cd}_{item.idx}:{item.party_id}"
            redis_client.set(redis_key, pickle.dumps(encoding))

def get_party_id_from_key(key):
    """Lấy party_id từ key Redis"""
    return key.split(":")[-1]

# Dùng asyncio để chạy song song việc so sánh
async def compare_face_with_redis(key, face_encoding):
    known_encoding = pickle.loads(redis_client.get(key))
    dist = cosine(face_encoding, known_encoding)
    return key.decode(), dist

async def add_dms_history(db, party_id, geo_point_id, branch_id):
    current_timestamp = datetime.now()
    ## Kiểm tra xem đã có lịch sử trong 10 giây qua chưa
    if party_id in history and (current_timestamp - history[party_id]).total_seconds() < 10:
        return "Already added"
    
    entity = DmsPartyLocationHistoryCreate()
    entity.party_id = party_id
    entity.geo_point_id = str(geo_point_id)
    entity.note = "Camera detection"
    entity.source_timekeeping = "Camera detection"
    entity.branch_id = str(branch_id)
    entity.created_date = current_timestamp
    entity.updated_date = current_timestamp
    entity.created_stamp =current_timestamp
    entity.created_tx_stamp = current_timestamp
    entity.last_updated_stamp = current_timestamp
    entity.last_updated_tx_stamp = current_timestamp
    try:
        DmsPartyLocationHistoryStore.create_location_history(db, entity)
        history[party_id] = current_timestamp
        return "Added"
    except Exception as e:
        print("An error occurred while adding DMS history:", e)
        return "Error"


@app.get("/reload-cache")
async def reload_cache():
    await load_employee_data()
    return {"message": "Reloaded face data"}

@app.on_event("startup")
async def startup_event():
    await load_employee_data()

@app.post("/recognize")
async def recognize_face(request: Request):
    try:
        """API nhận diện khuôn mặt"""
        form = await request.form()  # Lấy toàn bộ form data
        tenant_cd = form.get("tenant_cd")
        geo_point_id = form.get("geo_point_id")
        branch_id = form.get("branch_id")
        
        if not tenant_cd:
            return {"error": "Database name is required."}
        
        results = []
        db = next(get_db(tenant_cd))
        # Lấy danh sách tất cả các key từ Redis một cách bất đồng bộ
        redis_keys = list(redis_client.keys(f"{tenant_cd}_*"))
        for key in form.keys():
            if key.startswith("image"):
                image = form[key]
                
                image_bytes = await image.read()
                image = np.frombuffer(image_bytes, dtype=np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
                # Chuyển ảnh sang RGB
                # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Trích xuất đặc trưng khuôn mặt
                face_encoding = embedder.embeddings([image])[0]
        
                # Chạy song song các tác vụ nhận diện
                comparisons = await asyncio.gather(*[compare_face_with_redis(key, face_encoding) for key in redis_keys])

                # Tìm kết quả tốt nhất
                best_match, min_dist = min(comparisons, key=lambda x: x[1])
                if min_dist < 0.3:  # Ngưỡng nhận diện
                    party_id = get_party_id_from_key(best_match)
                    added = await add_dms_history(db, party_id, geo_point_id, branch_id)  # Thêm lịch sử vào DB
                    results.append(RecorgnizeFace(party_id, min_dist, added))
                else:
                    results.append(RecorgnizeFace("Unknown", min_dist, "Not added"))
        return JSONResponse(status_code=status.HTTP_200_OK, content=[{"party_id": result.party_id, "score": result.score, "added": result.added} for result in results])
    except Exception as e:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": f"Error recognizing face: {e}"})

@app.post("/add_employee")
async def add_employee(request: Request):
    """
    Endpoint để thêm nhân viên mới.
    Client gửi các file hình với các key "image0", "image1", ... và dữ liệu form "name".
    Server sẽ:
      1. Lấy tên nhân viên từ form.
      2. Duyệt qua các key bắt đầu bằng "image" để đọc nội dung file.
      3. Decode ảnh, chuyển sang RGB và trích xuất encoding từ FaceNet.
      4. Lưu encodings và lưu vào MySQL cũng như cập nhật cache Redis.
    """
    form = await request.form()  # Lấy toàn bộ form data
    party_id = form.get("party_id")
    tenant_cd = form.get("tenant_cd")

    if not party_id or not tenant_cd:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "Some required fields are missing."})

    try:
        db = next(get_db(tenant_cd))

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
                encoding_blob = pickle.dumps(encoding)
                entity = PersonEmbeddingCreate(party_id=party_id, embedding=encoding_blob)
                new_embedding = PersonEmbeddingStore.create_person_embedding(db, entity)
                redis_key = f"{tenant_cd}_{new_embedding.idx}:{party_id}"
                redis_client.set(redis_key, encoding_blob)
        
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": f"Added employee {party_id}"})
    except Exception as e:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": f"Error adding employee: {e}"})
    
# @app.post("/add_employee")
# async def add_employee(request: Request):
#     """
#     Endpoint để thêm nhân viên mới.
#     Client gửi các file hình với các key "image0", "image1", ... và dữ liệu form "name".
#     Server sẽ:
#       1. Lấy tên nhân viên từ form.
#       2. Duyệt qua các key bắt đầu bằng "image" để đọc nội dung file.
#       3. Decode ảnh, chuyển sang RGB và trích xuất encoding từ FaceNet.
#       4. Tính trung bình encoding từ các ảnh và lưu vào MySQL cũng như cập nhật cache Redis.
#     """
#     form = await request.form()  # Lấy toàn bộ form data
#     party_id = form.get("party_id")
#     tenant_cd = form.get("tenant_cd")

#     if not party_id or not tenant_cd:
#         return {"error": "Some required fields are missing."}
    
#     encodings = []

#     # Duyệt qua các key của form data
#     for key in form.keys():
#         if key.startswith("image"):
#             file = form[key]  # file là UploadFile
#             image_bytes = await file.read()
#             image_np = np.frombuffer(image_bytes, dtype=np.uint8)
#             img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
#             if img is None:
#                 continue
#             # Chuyển ảnh sang RGB (FaceNet thường yêu cầu ảnh RGB)
#             rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             # Trích xuất encoding, giả sử mỗi ảnh có đúng 1 khuôn mặt
#             encoding = embedder.embeddings([rgb_image])[0]
#             encodings.append(encoding)
    
#     if not encodings:
#         return {"error": "Không nhận được hình hợp lệ để trích xuất khuôn mặt."}
    
#     # Lưu vào database
#     try:
#         db = next(get_db(tenant_cd))
#         exist = PersonEmbeddingStore.get_person_embedding_by_id(db, party_id)
        
#         if exist:
#             existing_blob = pickle.loads(exist.embedding)
#             encodings.append(existing_blob)

#         avg_encoding = np.mean(encodings, axis=0)
#         encoding_blob = pickle.dumps(avg_encoding)

#         entity = PersonEmbeddingCreate(party_id=party_id, embedding=encoding_blob)
#         if exist:
#             PersonEmbeddingStore.update_person_embedding(db, party_id, entity)
#         else:
#             PersonEmbeddingStore.create_person_embedding(db, entity)


#     except Exception as e:
#         return {"error": f"Lỗi khi lưu vào MySQL: {e}"}
    
#     # Cập nhật cache Redis ngay lập tức
#     try:
#         redis_key = f"{tenant_cd}:{party_id}"
#         redis_client.set(redis_key, encoding_blob)
#     except Exception as e:
#         return {"error": f"Lỗi khi cập nhật Redis: {e}"}

#     return {"message": f"Added employee {party_id}"}

@app.delete("/delete_employee/{tenant_cd}/{party_id}")
async def delete_employee(tenant_cd:str, party_id: str):
    """
    Endpoint để xóa nhân viên.
    Client gửi dữ liệu form với các trường "party_id" và "tenant_cd".
    Server sẽ xóa nhân viên khỏi MySQL và Redis.
    """
    if not party_id or not tenant_cd:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": "Some required fields are missing."})
    
    try:
        db = next(get_db(tenant_cd))
        PersonEmbeddingStore.delete_person_embedding(db, party_id)
        redis_keys = redis_client.keys(f"{tenant_cd}_*")
        for key in redis_keys:
            if party_id == get_party_id_from_key(key.decode()):
                redis_client.delete(key)
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": f"Deleted employee {party_id}"})
    except Exception as e:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": f"Error deleting employee: {e}"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="trace")
