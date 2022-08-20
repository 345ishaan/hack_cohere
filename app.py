import streamlit as st
import numpy as np
import easyocr as ocr
from PIL import Image
from text_classifier import Cohere


st.header("Your personal document analyzer - Co:here application")

api_key = st.text_input("API Key:", type="password")        #text box for API key 

image = st.file_uploader("Upload your document here", type=['png', 'jpg', 'jpeg'])


@st.cache
def load_model(): 
    reader = ocr.Reader(['en'],model_storage_directory='.')
    return reader

reader = load_model()
cohere = Cohere(api_key)                                    #initialization CoHere
cohere.fill_examples()   

if image is not None:

    input_image = Image.open(image) #read image
    st.image(input_image) #display image

    with st.spinner("🤖 Running Analyzer! "):
        result = reader.readtext(np.array(input_image))
        result_text = []
        for res in result:
            result_text.append(res[1])
        all_text = ' '.join(result_text)
        st.write(result_text)
        here = cohere.classify([all_text])[0]                  #prediction 
        col1, col2 = st.columns(2)
        for no, con in enumerate(here.confidence):              #display likelihood for each label
            if no % 2 == 0:                                     # in two columns
                col1.write(f"{con.label}: {np.round(con.confidence*100, 2)}%")
                col1.progress(con.confidence)
            else:
                col2.write(f"{con.label}: {np.round(con.confidence * 100, 2)}%")
                col2.progress(con.confidence)    
else:
    st.write("Upload an Image")

