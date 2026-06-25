import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Configuration Styling
st.set_page_config(page_title="TMDB Movie Predictor", page_icon="🎬", layout="centered")

st.title("🎬 Movie Success & Rating Predictor Engine")
st.markdown("This machine learning system predicts a movie's success metrics using random forest regression model arrays.")

# 2. Safely Load the Trained Binary Model Brain
@st.cache_resource
def load_model():
    return joblib.load('movie_model.pkl')

try:
    model = load_model()
    
    st.info("💡 Enter your production parameters below to predict the rating outcome.")
    
    # 3. Create Clean Numerical Form Fields for the UI
    col1, col2 = st.columns(2)
    
    with col1:
        movie_id = st.number_input("Movie ID Reference Registration:", min_value=1, max_value=999999, value=550)
    with col2:
        vote_count = st.number_input("Expected Vote Count Metrics:", min_value=0, max_value=50000, value=1200)

    # 4. Process Prediction Actions (UPDATED WITH PANDAS DATAFRAME DETECTED LABELS)
    if st.button("Calculate Predictive Analysis", type="primary"):
        # We use pd.DataFrame matching your exact original dataset feature columns ('id' and 'vote_count')
        input_data = pd.DataFrame([[movie_id, vote_count]], columns=['id', 'vote_count'])
        
        # Run calculation
        prediction = model.predict(input_data)[0]
        st.success(f"📊 Predicted Vote Average Rating Score: **{prediction:.2f} / 10**")
        
        # Add a helpful validation display bar
        st.progress(float(prediction / 10.0))

except FileNotFoundError:
    st.error("Missing File Error: Please verify 'movie_model.pkl' is uploaded in the directory layer.")
