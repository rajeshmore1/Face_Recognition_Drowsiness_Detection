from pandas.core.frame import DataFrame
import streamlit as st
import cv2
from PIL import Image
from Recognition import face_identifier
import pandas as pd

def main():
    """ Face Recognition App"""
    st.title("Face Recognition")
    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Face Recognition WebApp</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Original Image")
        st.image(our_image)
    
    identifier = face_identifier() 
    @st.cache
    def recog_system(img):
        result_img, person_name = identifier.recogniser(img)
        return result_img, person_name
    
    @st.cache
    def drowsy_detector(img):
        result_img, drowsiness_status = identifier.drowsiness_detection(img)
        return result_img, drowsiness_status

    
    

    if st.button("Recognise"):
        recog_img, person_name = recog_system(our_image)
        st.image(recog_img)

    if st.button('Check Drowsiness'):
        result_img, drowsiness_status = drowsy_detector(our_image)
        st.image(result_img)


    if st.button('Preview Attendance'):
        recog_img, person_name = recog_system(our_image)
        result_img, drowsiness_status = drowsy_detector(our_image)
        identifier.MarkAttendance(person_name, drowsiness_status)
        data = pd.read_csv(r'C:\Users\Rohan\Desktop\face detection\attendance.csv')
        st.write(data)


if __name__ == '__main__':
   main()