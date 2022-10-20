import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import warnings
warnings.simplefilter(action='ignore')

st.set_page_config(page_title="Face Orientation Predictor",layout="wide")

x={0:'Upright', 1:'Rotated Clockwise',2:'Rotated Anti-Clockwise', 3:'Upside Down'}

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

def convert(image):
    nimg=[]
    im = Image.open(image)
    im=im.resize((64,64))
    im3=np.array(im)
    nimg.append(im3)
    nimg=np.array(nimg)
    return nimg

model=load_model()


st.title("Orientation Predictor")


image = st.file_uploader(label="Upload and image",type=['png','jpg','jpeg'])
if image is not None:
    nimg=convert(image)
    preds=model.predict(nimg,verbose=0)
    out=np.argmax(preds)
    st.title(x[out])
    st.image(image=image)
    image = Image.open(image)
    image=image.resize((256,256))
    if x[out]=="Rotated Clockwise":
        image=image.transpose(Image.ROTATE_90)
    elif x[out]=="Rotated Anti-Clockwise":
        image=image.transpose(Image.ROTATE_180).transpose(Image.ROTATE_90)
    elif x[out]=="Upside Down":
        image=image.transpose(Image.ROTATE_180)
    st.image(image=image)
    

    