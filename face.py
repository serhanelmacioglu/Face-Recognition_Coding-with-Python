import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep


def get_encoded_faces():

    encoded = {}

    for dirpath, dnames, fnames in os.walk("./face_repository"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("face_repository/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded


def unknown_image_encoded(img):

    face = fr.load_image_file("face_repository/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding


def classify_face(im):
 
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)
 
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
   
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            
            cv2.rectangle(img, (left-20, top-10), (right+20, bottom+15), (300, 0, 0), 2)

       
            cv2.rectangle(img, (left-20, bottom -10), (right+20, bottom+15), (500, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left -10, bottom + 10), font, 0.5, (300, 300, 300), 1)


   
    while True:

        cv2.imshow('Whom are you looking for?', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return face_names 


print(classify_face("test1.jpg")) # You can either try to find people "test2.jpg" or "test1.jpg" in the string.
