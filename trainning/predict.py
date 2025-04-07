import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import yaml


# Load the pre-trained model
model = load_model('model.h5')
IMAGE_SIZE = (160, 160) 

# Use Mediapipe for face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
# Start video capture
cap = cv2.VideoCapture(0)

# Define labels for predictions (update according to your model's output)
labels = None
with open('class_labels.yaml', 'r') as file:
    labels = yaml.safe_load(file)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the frame to RGB as Mediapipe works with RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            # Get bounding box coordinates
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Extract the face region
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue

            # Preprocess the face for the model
            face_resized = cv2.resize(face, IMAGE_SIZE)  # Adjust size based on your model's input
            face_normalized = face_resized / 255.0
            face_reshaped = np.expand_dims(face_normalized, axis=0)

            # Predict using the model
            predictions = model.predict(face_reshaped)
            accuracy = str(np.max(predictions) * 100)[:2] + "%"

            if labels is not None:
                label_index = np.argmax(predictions)
                label = labels[label_index]
                cv2.putText(frame, label + accuracy, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()