import streamlit as st
import numpy as np

from text_classifier import Cohere


st.header("Your personal text classifier - Co:here application")

api_key = st.text_input("API Key:", type="password")        #text box for API key 

description = [st.text_input("Description:")]               #text box for text to predict

cohere = Cohere(api_key)                                    #initialization CoHere
cohere.fill_examples()                                   #loading training set 

if st.button("Classify"):   
    here = cohere.classify(description)[0]                  #prediction 
    col1, col2 = st.columns(2)
    for no, con in enumerate(here.confidence):              #display likelihood for each label
        if no % 2 == 0:                                     # in two columns
            col1.write(f"{con.label}: {np.round(con.confidence*100, 2)}%")
            col1.progress(con.confidence)
        else:
            col2.write(f"{con.label}: {np.round(con.confidence * 100, 2)}%")
            col2.progress(con.confidence)