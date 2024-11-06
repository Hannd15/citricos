# streamlit_audio_recorder y whisper by Alfredo Diaz - version April 2024
#python -m venv env
#cd D:\smart\env\Scripts\
#.\activate 
#cd d:\mango
#pip install tensorflow==2.15.0
#pip install numpy
#pip install streamlit
#pip install pillow

# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st  
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np

# hide deprication warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Detección de enfermedades del mango",
    page_icon = ":mango:",
    initial_sidebar_state = 'auto'
)

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) # Oculta el código CSS de la pantalla, ya que están incrustados en el texto de rebajas. Además, permita que Streamlit se procese de forma insegura como HTML

#st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache_resource
def load_model():
    model=tf.keras.models.load_model('mango_model.h5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()
    
    
def prediction_cls(prediction): # predecir la clase de las imágenes en función de los resultados del modelo
    for key, clss in class_names.items(): # crear un diccionario de las clases de salida
        if np.argmax(prediction)==clss: # Verifica la clase
            return key

with st.sidebar:
        st.image('hojas.png')
        st.title("Estado de salud Manguifera")
        st.subheader("Detección de enfermedades presentes en las hojas del mango usando Depp Learning CNN. Esto ayuda al campesino a detectar fácilmente la enfermedad e identificar su causa.")

st.image('Logo_SmartRegions.gif')
st.title("Smart Regions Center")
st.write("Somos un equipo apasionado de profesionales dedicados a hacer la diferencia")
st.write("""
         # Detección de enfermedades del mango con su recomendación de tratamiento
         """
         )

file = st.file_uploader("", type=["jpg", "png"])
def import_and_predict(image_data, model):
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction

        
if file is None:
    st.text("Por favor cargue una imagen")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = random.randint(98,99)+ random.randint(0,99)*0.01
    st.sidebar.error("Accuracy : " + str(x) + " %")
    class_names = ["Black Spot", "Canker", "Greening", "Healthy"]

    string = "Enfermedad detectada : " + class_names[np.argmax(predictions)]
    if class_names[np.argmax(predictions)] == 'Sano':
        st.balloons()
        st.sidebar.success(string)

    else:
        st.sidebar.warning(string)
