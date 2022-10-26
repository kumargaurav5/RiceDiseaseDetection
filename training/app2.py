import streamlit as st
import tensorflow as tf
import streamlit as st
import cv2



@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('rice.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Rice Leaf Disease Classification
         """
         )

file = st.file_uploader("Please upload an Leaf Image file", type=["jpg", "png"])

import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
        data = np.ndarray(shape=(1,256,256,3),dtype=np.float32)
        size = (256,256)
        image = ImageOps.fit(image_data,size,Image.ANTIALIAS)
        image_array=np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32))
        data[0]=normalized_image_array
        prediction = model.predict(data)
        #print(prediction)
        np.argmax(prediction)
        return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image,model)
    score = tf.nn.softmax(predictions[0])
    st.write(predictions)
    st.write(score)
    class_names = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    print(predicted_class)
    st.write(predicted_class)
    st.write(confidence)
    #print(
    #"This image most likely belongs to {} with a {:.2f} percent confidence."
    #.format(class_names[np.argmax(predictions[0])], 100 * np.max(score))
#)


st.write("   Made By Gaurav & Pranshu ")
