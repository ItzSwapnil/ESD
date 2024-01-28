import cv2
import face_recognition

# Load the image
image = cv2.imread("D:/Projects/AISC/img/lib/Swapnil.jpg")

# Find the face locations and encodings
face_locations = face_recognition.face_locations(image)
face_encodings = face_recognition.face_encodings(image, face_locations)

# Print the face encodings
print(face_encodings)