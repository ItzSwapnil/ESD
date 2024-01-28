import cv2
import face_recognition

# webcam
video_capture = cv2.VideoCapture(0)

# Capture a frame from the webcam
ret, frame = video_capture.read()

# Find the face locations and encodings
face_locations = face_recognition.face_locations(frame)
face_encodings = face_recognition.face_encodings(frame, face_locations)

# Print the face encodings
print(face_encodings)

# Close the program
video_capture.release()