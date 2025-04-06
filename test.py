import redis
import pickle
import cv2
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

embedder = FaceNet()  # Load model FaceNet

redis_client = redis.Redis(host='localhost', port=6379, db=0)

image = cv2.imread('test.jpg')
# Chuyển ảnh sang RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Trích xuất đặc trưng khuôn mặt
face_encoding = embedder.embeddings([rgb_image])[0]

# So sánh với dữ liệu trong Redis
min_dist = 1.0
best_match = "Unknown"
print(redis_client.keys())
for key in redis_client.keys():
    print(key)
    known_encoding = pickle.loads(redis_client.get(key))
    dist = cosine(face_encoding, known_encoding)
    print(dist)
    if dist < min_dist:
        min_dist = dist
        best_match = key.decode()

print(best_match)