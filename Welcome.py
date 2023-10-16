import streamlit as st
import pandas as pd
import numpy as np
import joblib 

st.title(' Mall Custmer Segmentation Project')

def get_inputs():
    
    #get user inputs
    lists = []
    col_names = ['Gender' , 'Age', 'Annual Income (k$)','Spending Score (1-100)']
    
    Gender = st.text_input("Insert Gender")
    Age = st.number_input('Insert Age')
    Income = st.number_input('Insert Annual Income')
    Score = st.number_input('Insert Spending Score [1-100]')
    
    lists.append(Gender)
    lists.append(Age)
    lists.append(Income)
    lists.append(Score)
    exo = pd.DataFrame(lists)
    exo = exo.transpose()
    exo.columns = col_names
    st.write('The entered values are ', Gender, Age,Income,Score) 
    st.write(exo) 
#create predict button:
    st.button("Reset", type="primary")
    if st.button('Predict'):
    	reconstructed_model = joblib.load('gmm_pipeline.joblib')        
    	pred= reconstructed_model.predict(exo)
    	st.subheader('Predicted Customer Cluster is: ', divider='rainbow')
    	st.subheader(pred)
    
  
   
  
   
    

get_inputs()


