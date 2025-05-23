from datetime import datetime
from fastapi import FastAPI, Request, status, Depends
from fastapi.security import OAuth2PasswordBearer
import uvicorn
import redis
import cv2
import numpy as np
from keras_facenet import FaceNet
import pickle
from scipy.spatial.distance import cosine
import asyncio
import json, base64
from consts import TENANT_DICT, THRESHOLD, OVERLAP_TIME
from api import load_all_embeddings, checkInByFaceRecognition, create_person_embedding, delete_person_embedding
from logger_service import LoggerService
from pydantic import BaseModel
from typing import Optional, Any

class RecorgnizeFace:
    def __init__(self, party_id: str, fullName: str):
        self.party_id = party_id
        self.fullName = fullName

class ResponseModel(BaseModel):
    status: int
    message: str = "OK"
    data: Optional[Any] = None
    
app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# Kết nối Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

embedder = FaceNet()  # Load model FaceNet

history = {}
LOGGER = LoggerService().get_logger()

# Hàm tải dữ liệu nhân viên từ DB vào cache
async def load_employee_data():
    redis_client.flushdb()  # Xóa cache cũ

    for tenant_cd in TENANT_DICT.keys():
        try:
            data = await load_all_embeddings(tenant_cd)  # Tải dữ liệu từ DB vào cache
            for item in data:
                embedding_bytes = base64.b64decode(item.get('embedding'))
                redis_key = f"{tenant_cd}_{item.get('id')}:{item.get('partyId')}"
                redis_client.set(redis_key, embedding_bytes)
        except Exception as e:
            LOGGER.error(f"Loading data for [{tenant_cd}]: {e}")

def get_party_id_from_key(key):
    """Lấy party_id từ key Redis"""
    return key.split(":")[-1]

# Dùng asyncio để chạy song song việc so sánh
async def compare_face_with_redis(key, face_encoding):
    known_encoding = pickle.loads(redis_client.get(key))
    dist = cosine(face_encoding, known_encoding)
    return key.decode(), dist

async def add_dms_history(tenant_cd, party_id, address, branch_id, image_bytes, token):
    current_timestamp = datetime.now()
    ## Kiểm tra xem đã có lịch sử trong 10 giây qua chưa
    if party_id in history and (current_timestamp - history[party_id]['current_timestamp']).total_seconds() < OVERLAP_TIME:
        return history[party_id]["fullName"]
    try:
        fullName = await checkInByFaceRecognition(tenant_cd, party_id, address, branch_id, image_bytes, token)
        history[party_id] = {
            "fullName": fullName,
            "current_timestamp": current_timestamp,
        }
        LOGGER.info(f"Added timekeeping for [{party_id}]")
        return fullName
    except Exception as e:
        LOGGER.error(f"Adding timekeeping for [{party_id}]: {e}")
        return "Unknown"

@app.get("/reload-cache")
async def reload_cache():
    try:
        await load_employee_data()
        LOGGER.info("Reload cache")
        return ResponseModel(status=status.HTTP_200_OK, data="Reloaded face data")
    except Exception as e:
        LOGGER.error(e)
        return ResponseModel(status=status.HTTP_500_BAD_REQUEST, message=e)

@app.on_event("startup")
async def startup_event():
    await load_employee_data()

"""API nhận diện khuôn mặt"""
@app.post("/recognize")
async def recognize_face(request: Request, token: str = Depends(oauth2_scheme)):
    try:
        form = await request.form()  # Lấy toàn bộ form data
        tenant_cd = form.get("tenant_cd")
        address_str = form.get("address")
        address = json.loads(address_str)[0] if address_str else {}
        branch_id = form.get("branch_id")
            
        redis_keys = list(redis_client.keys(f"{tenant_cd}_*"))
        if not redis_keys or len(redis_keys) == 0:
            LOGGER.warning("No employee data found in Redis.")
            return ResponseModel(status=status.HTTP_400_BAD_REQUEST, message="No employee data found in Redis.")
        if not tenant_cd:
            LOGGER.warning("Tenant code is missing.")
            return ResponseModel(status=status.HTTP_400_BAD_REQUEST, message="Tenant code is missing.")
        
        results = []
        # Lấy danh sách tất cả các key từ Redis một cách bất đồng bộ
        for key in form.keys():
            if key.startswith("image"):
                image = form[key]
                
                image_bytes = await image.read()
                image = np.frombuffer(image_bytes, dtype=np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
                # Chuyển ảnh sang RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Trích xuất đặc trưng khuôn mặt
                face_encoding = embedder.embeddings([rgb_image])[0]
        
                # Chạy song song các tác vụ nhận diện
                comparisons = await asyncio.gather(*[compare_face_with_redis(key, face_encoding) for key in redis_keys])

                # Tìm kết quả tốt nhất
                best_match, min_dist = min(comparisons, key=lambda x: x[1])

                if min_dist < THRESHOLD:  # Ngưỡng nhận diện
                    party_id = get_party_id_from_key(best_match)
                    fullName = await add_dms_history(tenant_cd, party_id, address, branch_id, image_bytes, token)  # Thêm lịch sử vào DB
                    results.append(RecorgnizeFace(party_id, fullName))
                else:
                    results.append(RecorgnizeFace("", "Unknown"))
                    # Tìm kết quả tốt nhất

        return ResponseModel(status=status.HTTP_200_OK, data=[result.__dict__ for result in results])
    except Exception as e:
        LOGGER.error(f"Recognizing face: {e}")
        return ResponseModel(status=status.HTTP_500_INTERNAL_SERVER_ERROR, message=f"Error recognizing face: {e}")

"""
    Endpoint để thêm nhân viên mới.
    Client gửi các file hình với các key "image0", "image1", ... và dữ liệu form "name".
    Server sẽ:
        1. Lấy mã nhân viên từ form.
        2. Duyệt qua các key bắt đầu bằng "image" để đọc nội dung file.
        3. Decode ảnh, chuyển sang RGB và trích xuất encoding từ FaceNet.
        4. Lưu encodings và lưu vào MySQL cũng như cập nhật cache Redis.
"""
@app.post("/add_employee")
async def add_employee(request: Request, token: str = Depends(oauth2_scheme)):
    form = await request.form()  # Lấy toàn bộ form data
    party_id = form.get("party_id")
    tenant_cd = form.get("tenant_cd")
        
    if not party_id or not tenant_cd:
        LOGGER.warning("Party ID or tenant code is missing.")
        return ResponseModel(status=status.HTTP_400_BAD_REQUEST, message="Party ID or tenant code is missing.")

    try:
        # Duyệt qua các key của form data
        data = []
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
                data.append({
                    "partyId": party_id,
                    "embedding": base64.b64encode(encoding_blob).decode('utf-8'),
                })
        if not data:
            LOGGER.warning("There is no valid image to extract face.")
            return ResponseModel(status=status.HTTP_400_BAD_REQUEST, message="There is no valid image to extract face")
        
        # Lưu vào database
        results =  await create_person_embedding(tenant_cd, data, token)
        for result in results:
            redis_key = f"{tenant_cd}_{result.get('id')}:{result.get('partyId')}"
            redis_client.set(redis_key, base64.b64decode(result.get('embedding')))
        
        LOGGER.info(f"Added employee {party_id} with {len(data)} images.")
        return ResponseModel(status=status.HTTP_200_OK, data=f"Added employee {party_id}")
    except Exception as e:
        LOGGER.error(f"Adding employee: {e}")
        return ResponseModel(status=status.HTTP_500_INTERNAL_SERVER_ERROR, message=e)


"""
    Endpoint để xóa đặc trưng của nhân viên.
    Client gửi dữ liệu form với các trường "party_id" và "tenant_cd".
    Server sẽ xóa đặc trưng của nhân viên khỏi MySQL và Redis.
    """
@app.delete("/delete_employee/{tenant_cd}/{party_id}")
async def delete_employee(tenant_cd:str, party_id: str, token: str = Depends(oauth2_scheme)):
    if not party_id or not tenant_cd:
        LOGGER.warning("Party ID or tenant code is missing.")
        return ResponseModel(status=status.HTTP_400_BAD_REQUEST, message="Party ID or tenant code is missing.")
    
    try:
        await delete_person_embedding(tenant_cd, party_id, token)
        redis_keys = redis_client.keys(f"{tenant_cd}_*")
        for key in redis_keys:
            if party_id == get_party_id_from_key(key.decode()):
                redis_client.delete(key)

        LOGGER.info(f"Deleted {party_id} from tenant {tenant_cd}.")
        return ResponseModel(status=status.HTTP_200_OK, message=f"Deleted employee {party_id}")
    except Exception as e:
        LOGGER.error(f"Error deleting employee: {e}")
        return ResponseModel(status=status.HTTP_500_INTERNAL_SERVER_ERROR, message=f"Error deleting employee: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="trace")
