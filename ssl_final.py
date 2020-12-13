import cv2
import streamlit as st
import os
import time
import mediapipe as mp
import PIL as Image

#TRANSLATE
from FDK.src.core.translate import Translator_M
from FDK.src.core.utils import utils
from pprint import pprint
import matplotlib.pyplot as plt
import numpy
from PIL import Image

#Make Directory
if not os.path.isdir("temp_img"):
    os.mkdir("temp_img")

#Page Config
st.set_page_config(
    page_title="SSL",
    page_icon=":hand:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.header("Shaftbury Sign Language")
st.info("Translates Sign Language for Everyone :smile:")

st.markdown('<hr>', unsafe_allow_html=True)

#Play Video
import cv2
import mediapipe as mp
import streamlit as st
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
y=["image"]

temp_status = len(os.listdir("image"))

#img = None

uploaded_file = st.file_uploader("Alternatively upload a file")
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img.save("image/hands.jpg")
    st.success("Uploaded.")

# For static images:
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5)

st.write("Raw image") 

try:
    st.image(img, use_column_width=True)
except NameError:
    pass

#Covert PIL to cv2
idx = 0
open_cv_image = numpy.array(img)
img = open_cv_image[:, :, ::-1].copy() 
image=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

results = hands.process(image)

image_hight, image_width, _ = image.shape
annotated_image = image.copy()
for hand_landmarks in results.multi_hand_landmarks:
  idx = idx+1
  mp_drawing.draw_landmarks(
      annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
outimage=cv2.imwrite("temp_img/" +
      str(idx) + '.png', cv2.cvtColor(annotated_image,cv2.COLOR_BGR2RGB))

st.write("Output")
st.image("temp_img/1.png",use_column_width=True )

hands.close()




 
 

#for idx, file in enumerate(temp_img):
  # Read an image, flip it around y-axis for correct handedness output (see
  # above).


#Translator
st.subheader("Translate to your Language!")  

Trans = Translator_M(task='Helsinki-NLP/opus-mt-en-ROMANCE')

text = st.text_input("Text to Translate") 
language = st.multiselect("Language",["French","Spanish","Portugese"])
lg = ""



if not text or not language:
  st.empty()

else:  
  language=language[0]

  if language == "French":
    lg = "fr"
  elif language == "Spanish":
    lg = "es"
  elif language == "Portugese":
    lg = "pt"  

  
  text_ = ">>{}<< {}".format(lg,text)
  
  text_to_translate = [text_]

  st.write("Translated Text: ")
  
  output = Trans.predict(text_to_translate)
  st.write(output)  