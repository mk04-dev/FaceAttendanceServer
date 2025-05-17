import cv2
from numpy.linalg import norm
import numpy as np
import consts
from deepface import DeepFace
from mtcnn import MTCNN
from consts import TENANT_DICT
detector = MTCNN()  # Bộ phát hiện khuôn mặt

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # Bộ phát hiện khuôn mặt

def extract_face(image):
    faces = detector.detect_faces(image)

    if len(faces) == 0:
        return None
    
    x, y, w, h = faces[0]['box']
    face = image[y:y+h, x:x+w]  # Cắt vùng mặt
    face = cv2.resize(face, (160, 160))  # Resize theo yêu cầu của FaceNet
    # face = np.asarray(face, dtype=np.float32)
    # face = (face - 127.5) / 127.5  # Chuẩn hóa dữ liệu
    return face

def detect_face(frame):
    """Kiểm tra xem có khuôn mặt trong ảnh không"""
    # faces = DeepFace.extract_faces(frame, detector_backend='opencv', enforce_detection=False)
    # return faces

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for face in faces:
        (x, y, w, h) = face
        cv2.rectangle(frame, (x, y), (x+w, y+h), consts.COLOR_GREEN, consts.THICKNESS)
    return faces  # Trả về danh sách khuôn mặt

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def get_embedding(img):
    """Lấy embedding của khuôn mặt"""
    return DeepFace.represent(img, model_name="Facenet", enforce_detection=False)[0]['embedding']

def get_tenant_info(tenant_id):
    """Lấy thông tin tenant từ TENANT_DICT"""
    if tenant_id not in TENANT_DICT:
        raise ValueError(f"Tenant ID [{tenant_id}] not found")
    return TENANT_DICT[tenant_id]