import streamlit as st
import numpy as np
import easyocr as ocr
from PIL import Image
from text_classifier import Cohere


st.header("Your personal document analyzer - Co:here application")

# api_key = st.text_input("API Key:", type="password")        #text box for API key 

image = st.file_uploader("Upload your document here", type=['png', 'jpg', 'jpeg'])


@st.cache
def load_model(): 
    reader = ocr.Reader(['en'],model_storage_directory='.')
    return reader

reader = load_model()
cohere = Cohere()
cohere.fill_examples()   

if image is not None:

    input_image = Image.open(image) #read image
    st.image(input_image) #display image

    with st.spinner("🤖 Running Analyzer! "):
        result = reader.readtext(np.array(input_image))
        valid_words = []
        for res in result:
            if res[2] > 0.5:
                text = res[1].lower()
                text = text.strip()
                strip = ''.join(c for c in text if c.isalpha())
                if len(strip) > 2:
                    valid_words.append(strip)
        result_text = ' '.join(valid_words)
        st.write(result_text)
        here = cohere.classify([result_text])[0]
        col1, col2 = st.columns(2)
        for no, con in enumerate(here.confidence):
            if no % 2 == 0:
                col1.write(f"{con.label}: {np.round(con.confidence*100, 2)}%")
                col1.progress(con.confidence)
            else:
                col2.write(f"{con.label}: {np.round(con.confidence * 100, 2)}%")
                col2.progress(con.confidence)    
else:
    st.write("Upload an Image")
