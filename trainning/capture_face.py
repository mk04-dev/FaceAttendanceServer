"""
This script captures images from a webcam and saves them to a specified directory.
It uses Mediapipe for face detection and allows the user to start and stop the image collection process.
Press 'a' to start collecting images for a specific employee ID, and 's' to stop.
Press 'q' to quit the program.
"""

import cv2
import mediapipe as mp
import os

IMG_SIZE = (160, 160)  # Size to which the images will be resized

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

collecting = False
emp_id = None
count = 0
interval = 30  # frames between captures
current_frame = 0
output_dir = "./face-recognition-dataset"
# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for face detection
    results = face_detection.process(rgb_frame)

    current_frame += 1
    # Draw face detection annotations on the frame
    if results.detections:
        for detection in results.detections:
            if collecting:
                # Extract the bounding box coordinates
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                
                if current_frame % interval == 0:
                    # Save the face image
                    face_image = frame[y:y+h, x:x+w]
                    face_image = cv2.resize(face_image, IMG_SIZE)  # Resize to the required size
                    cv2.imwrite(f"{output_dir}/{emp_id}/{emp_id}_{count}.jpg", face_image)
                    count += 1
                cv2.putText(frame, f"Collected {count}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            mp_drawing.draw_detection(frame, detection)

    # Display the frame
    cv2.imshow('Face Detection', frame)
    
    key = cv2.waitKey(1) & 0xFF
    # Break the loop on 'q' key press
    if key == ord('q'):
        break
    elif key == ord('a'):
        emp_id = input("Enter employee ID: ").strip()
        os.makedirs(f"./{output_dir}/{emp_id}", exist_ok=True)
        collecting = True
    elif key == ord('s') and collecting:
        # Save the frame as an image
        collecting = False
        emp_id = None
        count = 0
        print("Stopped collecting frames.")
        
# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()