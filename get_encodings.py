import cv2
import face_recognition
import numpy as np
import os
import pickle

path = r'C:\Users\Rohan\Desktop\face detection\KnownFaces'
images = []
names = []
encode_list = []
mylist = os.listdir(path)
for img in mylist:
    cur_img = cv2.imread(f'{path}/{img}')
    images.append(cur_img)
    names.append(os.path.splitext(img)[0])

for img in images:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encode = face_recognition.face_encodings(img)[0]
    encode_list.append(encode)

encodings_dict = dict()
encodings_dict['encodings'] = encode_list
encodings_dict['names'] = names

with open(r'C:\Users\Rohan\Desktop\face detection\encodings.pickle', 'wb') as f:
        pickle.dump(encodings_dict, f)
