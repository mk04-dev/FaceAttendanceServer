import time
import pickle
import threading
import cv2
import faiss
import numpy as np
from database import SessionLocal
import consts
from schemas import PersonEmbeddingCreate
from utils import detect_face, get_embedding
from repositories.person_embedding_store import PersonEmbeddingStore
from deepface import DeepFace
import concurrent.futures

# Cấu hình OpenCV tối ưu
cv2.setUseOptimized(True)
cv2.setNumThreads(2)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)  # Tối đa 5 luồng song song

# Kết nối MySQL
db = SessionLocal()

# Load embeddings từ database lên RAM
def load_embeddings():
    data = PersonEmbeddingStore.get_all_person_embedding(db)
    employee_data = {}
    embeddings_list = []
    emp_ids = []
    for item in data:
        emp_ids.append(item.party_id)
        emb = pickle.loads(item.embedding)
        embeddings_list.append(np.array(emb, dtype=np.float32))
    
    index = faiss.IndexFlatL2(128)

    if len(embeddings_list) > 0:
        embeddings_np = np.array(embeddings_list, dtype=np.float32).reshape(-1, 128)
        # Dùng FAISS để tăng tốc tìm kiếm
        index.add(embeddings_np)
    else:
        print("⚠️ Không có embedding nào để thêm vào FAISS")
    return employee_data, index, emp_ids

employee_data, faiss_index, emp_ids = load_embeddings()

mode = consts.MODE_DETECT  # Mặc định là nhận diện

# Biến để thêm nhân viên mới
new_employee_id = None  # ID nhân viên mới
captured_images = []  # Lưu ảnh để lấy trung bình embedding
num_samples = consts.NUM_OF_SAMPLES  # Số lượng ảnh cần chụp

# Biến để kiểm soát cập nhật
last_recognized_id = None
last_recognized_time = 0
update_interval = 10  # Thời gian tối thiểu giữa 2 lần ghi (giây)

# Thread
recognition_result = None
recognition_lock = threading.Lock()

# Giảm lỗi nhận diện sai
id_buffer = []
buffer_size = 3  # Cần nhận diện 3 lần giống nhau trước khi lưu

cap = cv2.VideoCapture(0)

def extract_embedding(frame):
    emb = DeepFace.represent(frame, model_name='Facenet', enforce_detection=False)[0]['embedding']
    emb = np.array(emb, dtype=np.float32).reshape(1, -1)
    return emb


def search_emb(emb):
    D, I = faiss_index.search(emb, 1)
    print(D)
    if not D:
        return None
    return emp_ids[I[0][0]] if D[0][0] < 50 else None

def recognize_face(frame):
    # faces = DeepFace.extract_faces(frame, detector_backend='opencv', enforce_detection=False)
    # if len(faces) == 0:
    #     return None
    
    emb = DeepFace.represent(frame, model_name='Facenet', enforce_detection=False)[0]['embedding']
    emb = np.array(emb, dtype=np.float32).reshape(1, -1)
    D, I = faiss_index.search(emb, 1)
    print(D)
    if not D:
        return None
    return emp_ids[I[0][0]] if D[0][0] < 50 else None


def recognize_face_async(frame):
    start_time = time.time()
    future = executor.submit(extract_embedding, frame)
    def callback(fut):
        emb = fut.result()
        if emb is not None:
            emp_id = search_emb(emb)
            end_time = time.time()
            print(f"🕒 Thời gian nhận diện: {round(end_time - start_time, 2)} giây")

            if emp_id:
                print(f"✅ Nhận diện thành công: {emp_id}")
            else:
                print("❌ Không tìm thấy trong database!")
    future.add_done_callback(callback)
    # recognition_thread = threading.Thread(target=worker)
    # recognition_thread.start()
    # global recognition_result
    # emp_id = recognize_face(frame)
    # with recognition_lock:
    #     recognition_result = emp_id

def add_new_employee(employee_id, images):
    """Thêm nhân viên mới vào MySQL bằng cách lấy trung bình embedding"""
    embeddings = []
    for img in images:
        try:
            emb = get_embedding(img)
            embeddings.append(np.array(emb))
        except Exception as e:
            print(f"Lỗi khi trích xuất embedding: {e}")

    if embeddings:
        avg_embedding = np.mean(embeddings, axis=0).tolist()

        # Lưu vào MySQL
        embedding_bin = pickle.dumps(np.array(avg_embedding, dtype=np.float32))
        entity = PersonEmbeddingCreate(party_id=employee_id, embedding=embedding_bin)
        
        exist = PersonEmbeddingStore.get_person_embedding_by_id(db, employee_id)
        
        if exist:
            PersonEmbeddingStore.update_person_embedding(db, employee_id, entity)
        else:
            PersonEmbeddingStore.create_person_embedding(db, entity)

        # Cập nhật bộ nhớ
        # employee_data[employee_id] = np.array(embedding_bin, dtype=np.float32)
        faiss_index.add(np.array([avg_embedding], dtype=np.float32))
        print(f"✅ Đã thêm nhân viên {employee_id} với trung bình embedding!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (160, 160))
    faces = detect_face(small_frame)  # Phát hiện khuôn mặt
    if mode == "detect":
        if len(faces) > 0:
            recognize_face_async(small_frame)

    # elif mode == "register" and new_employee_id:
    #     cv2.putText(small_frame, f"Chế độ đăng ký - ID: {new_employee_id}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), consts.THICKNESS)
    #     cv2.putText(small_frame, f"Ảnh đã chụp: {len(captured_images)}/{num_samples}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), consts.THICKNESS)

    #     if len(faces) > 0:
    #         cv2.putText(small_frame, "Nhấn 'c' để chụp ảnh", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), consts.THICKNESS)

    #     if len(captured_images) >= num_samples:
    #         add_new_employee(new_employee_id, captured_images)
    #         mode = "detect"
    #         new_employee_id = None
    #         captured_images = []

    cv2.imshow("Face Recognition", small_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):  # Bắt đầu thêm nhân viên
        new_employee_id = str(input("Nhập ID nhân viên mới: "))
        mode = "register"
        captured_images = []
        print(f"🔴 Chế độ đăng ký. Nhấn 'c' để chụp ảnh...")
    elif key == ord('c') and mode == "register" and len(faces) > 0:  # Chụp ảnh nếu có khuôn mặt
        captured_images.append(small_frame.copy())
        print(f"📸 Ảnh {len(captured_images)}/{num_samples} đã được chụp!")
    elif key == ord('d'):  # Quay lại nhận diện
        mode = "detect"
        print("🔵 Chuyển về chế độ nhận diện")

cap.release()
cv2.destroyAllWindows()
