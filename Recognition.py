import cv2
import face_recognition
from scipy.spatial import distance as dist
import pandas as pd
from collections import OrderedDict
from imutils import face_utils
import numpy as np
import pickle
import os
import dlib
from datetime import datetime

class face_identifier:

    def __init__(self):

        with open(r'C:\Users\Rohan\Desktop\face detection\encodings.pickle', 'rb') as file:
            self.Knownencodesdict = pickle.load(file)


    def MarkAttendance(self, name, status):
        df = pd.read_csv(r'C:\Users\Rohan\Desktop\face detection\attendance.csv')
        now = datetime.now()
        date = now.strftime("%d-%m-%Y")
        time = now.strftime("%H:%M:%S")
        new_row = {'Name': name, 'Date' : date, 'Time' : time, 'Drowsiness_Status': status} 
        new_df = df.append(new_row, ignore_index=True)
        new_df.to_csv(r'C:\Users\Rohan\Desktop\face detection\attendance.csv', index = False)

    

    def recogniser(self, our_img):
        img = np.array(our_img.convert('RGB'))
        scale_img = cv2.resize(img, (0,0), None, 0.25, 0.25)
        scale_img = cv2.cvtColor(scale_img, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(scale_img)
        encodesCurframe = face_recognition.face_encodings(scale_img, facesCurFrame)
        

        names = self.Knownencodesdict['names']
        Knownencodes = self.Knownencodesdict['encodings']

        if len(facesCurFrame) == 0:
            print('No face Detected')
            return img, 'No face Detected'

        else:

            for encodesCurface, faceloc in zip(encodesCurframe, facesCurFrame):
                matches = face_recognition.compare_faces(Knownencodes, encodesCurface, tolerance= 0.5)
                faceDis = face_recognition.face_distance(Knownencodes, encodesCurface)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    person_name = names[matchIndex].upper()

                    (top, right, bottom, left) = faceloc
                    top, right, bottom, left =  top * 4, right * 4, bottom * 4, left * 4
                    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(img, (left, bottom + 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, person_name, (left + 6, bottom + 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 2)
                else:
                    person_name = 'Unknown'

                    (top, right, bottom, left) = faceloc
                    top, right, bottom, left =  top * 4, right * 4, bottom * 4, left * 4
                    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(img, (left, bottom + 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, person_name, (left + 6, bottom + 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 2)
            
            return img, person_name 
    def eye_aspect_ratio(self,eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])
        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        # return the eye aspect ratio
        return ear
    
  
    def drowsiness_detection(self, our_img):
        img = np.array(our_img.convert('RGB'))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_landmark_path = r'C:\Users\Rohan\Desktop\face detection\shape_predictor_68_face_landmarks.dat'
        detector = dlib.get_frontal_face_detector()
        detect = detector(gray, 1)
        if len(detect) == 0 :
            return 
        predictor = dlib.shape_predictor(face_landmark_path) 
        shape=predictor(img,detect[0])
        FACIAL_LANDMARKS_IDXS = OrderedDict([
            ("right_eye", (36, 42)),
            ("left_eye", (42, 48)),
        ])
        shape = face_utils.shape_to_np(shape)
        lStart, lEnd = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
        rStart, rEnd = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)
        
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

        EYE_AR_THRESH = 0.2
        Drowsiness_Status = True
        print(ear)
        if ear < EYE_AR_THRESH:
            cv2.putText(img, "DROWSINESS ALERT!", (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (225, 0, 0), 2)
        else:
            cv2.putText(img, "DROWSINESS NOT DETECTED!", (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 225, 0), 2)
            Drowsiness_Status = False
    
        return img, Drowsiness_Status









