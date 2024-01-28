import cv2
import os
import face_recognition
import time
import numpy as np

# encoded face
known_face_encodings = [
    #Swapnil
    np.array([-1.11389652e-01,  5.06371632e-02,  3.69762369e-02, -3.01712519e-03,
       -4.44842838e-02, -4.79151532e-02, -3.52991838e-03, -8.86070654e-02,
        1.94345325e-01, -1.43827155e-01,  2.17023641e-01, -2.10170001e-02,
       -1.86805367e-01, -1.36166692e-01,  4.08465192e-02,  9.43772346e-02,
       -1.71751246e-01, -1.32906750e-01, -1.99929737e-02, -9.86579061e-02,
       -1.96780520e-03,  2.76541039e-02,  3.73682678e-02,  8.49482566e-02,
       -1.85225643e-02, -3.80736470e-01, -1.27291918e-01, -1.60809770e-01,
        8.04972500e-02, -6.04524054e-02, -3.11600342e-02,  3.86456661e-02,
       -1.62052959e-01, -4.18956652e-02, -1.68678481e-02,  2.84161344e-02,
       -2.72323042e-02,  3.99893057e-03,  2.01917097e-01, -1.93110779e-02,
       -1.47843108e-01, -1.14248469e-01,  2.04273965e-03,  2.22106427e-01,
        1.33231789e-01,  6.89552873e-02, -2.62248199e-02, -1.47730513e-02,
        9.64023471e-02, -1.28896743e-01,  8.29665065e-02,  1.37513205e-01,
        1.91064745e-01,  3.20134573e-02,  8.02374333e-02, -1.67143553e-01,
       -2.04318985e-02,  4.44765352e-02, -2.02576682e-01,  8.07307586e-02,
       -1.31654628e-02, -1.46858945e-01, -1.76277198e-02,  2.21012102e-04,
        2.25026578e-01,  8.78311843e-02, -6.20264560e-02, -1.59321159e-01,
        1.75804839e-01, -2.07879752e-01,  1.58574805e-02,  1.62446663e-01,
       -8.66791755e-02, -1.82783186e-01, -2.24719360e-01,  2.58180872e-02,
        3.93337339e-01,  1.32592052e-01, -9.67823640e-02, -1.09569617e-02,
       -8.39497000e-02, -1.09149858e-01,  6.05649091e-02,  1.06246457e-01,
       -9.36810225e-02,  4.78416122e-03, -1.17772385e-01,  2.00153589e-02,
        1.00216582e-01, -8.05612095e-03, -7.18229488e-02,  1.89237714e-01,
       -7.20556155e-02,  9.15485546e-02,  2.12742761e-03, -1.26760378e-02,
       -4.74291518e-02,  3.63588110e-02, -2.91988812e-02, -1.46970255e-02,
        2.36677676e-02, -9.75785479e-02, -3.69907357e-03,  2.32678875e-02,
       -1.23982549e-01,  6.08927719e-02, -1.24242548e-02,  3.51332780e-03,
       -3.57622504e-02,  4.30837832e-02, -1.21800177e-01, -7.82424212e-02,
        1.45825237e-01, -2.90536284e-01,  1.72631353e-01,  1.37271345e-01,
       -1.30786244e-02,  2.08158344e-01, -2.63813473e-02,  1.05782785e-01,
       -2.93519720e-02, -1.56919122e-01, -1.54680938e-01, -2.27134526e-02,
        5.33659607e-02, -1.28160417e-02,  7.65092820e-02,  2.18688641e-02]) 
    ]
# known faces location
known_face_names = []
for image_file in os.listdir("C:/Users/Swapnil/PycharmProjects/ESD/img/lib"):
    image_path = os.path.join("C:/Users/Swapnil/PycharmProjects/ESD/img/lib", image_file)
    image = cv2.imread(image_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(image_file.split(".")[0])

# webcam
video_capture = cv2.VideoCapture(0)

# Variables
start_time = time.time()
current_user = None
unauthorized_user = None

while True:
    # Capture frames
    ret, frame = video_capture.read()

    # Find all the faces
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # No face = No output
    if len(face_encodings) == 0:
        current_user = None

    # Compare the faces
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

        # If the face matches, grant access
        if any(matches):
            current_user = known_face_names[matches.index(True)]
            start_time = time.time()

        # Capture Unauthorized user
        else:
            unauthorized_user = frame
            # 5sec Cooldown
            if time.time() - start_time > 5:
                unauthorized_user = None
            else:
                # Save unauthorized user's frame
                filename = f"unauthorized_user_{time.time()}.jpg"
                filepath = os.path.join("C:/Users/Swapnil/PycharmProjects/ESD/img/unauthorized_users", filename)
                cv2.imwrite(filepath, unauthorized_user)

    # If not authorized
    if current_user is None and unauthorized_user is not None:
        cv2.imshow('Unauthorized User', unauthorized_user)

    # If Authorized
    elif current_user is not None:
        print("Access granted to", current_user)
        cv2.imshow('Video', frame)

    # Resulting frame
    else:
        cv2.imshow('Video', frame)

    # 'Q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close
video_capture.release()
cv2.destroyAllWindows()