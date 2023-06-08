import face_recognition as fr
import cv2
import os
import numpy as np
import time

cap = cv2.VideoCapture(0)

def get_encoded_faces():
    encoded = {}

    for dirpath, dnames, fnames in os.walk('FaceRecognition/faces'):
        for f in fnames:
            if f.endswith('.png') or f.endswith('.jpg'):
                face = fr.load_image_file(os.path.join("FaceRecognition/faces", f))
                encoding = fr.face_encodings(face)[0]
                face_name = f.split(".")[0]
                encoded[face_name] = encoding
    
    return encoded


def classify_faces(encoded):
    known_face_encodings = []
    known_face_names = []

    for encoding in encoded.values():
        known_face_encodings.append(encoding)
    
    for fname in encoded.keys():
        known_face_names.append(fname)

    process_this_frame = True

    while True:
        ret, frame = cap.read()

        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = fr.face_locations(rgb_small_frame)
            face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = fr.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            if True in matches:
                match_index = matches.index(True)
                name = known_face_names[match_index]
        
            face_names.append(name)

        process_this_frame = not process_this_frame
        
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom -35 ), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) == ord('q'):
            break

classify_faces((get_encoded_faces()))

cap.release() 
cv2.destroyAllWindows()
