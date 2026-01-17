import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Page Configuration
st.set_page_config(page_title="Social Media Behavior Analysis", layout="centered")

# 2. Model aur Scaler load karein
try:
    model = pickle.load(open('cluster_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found! Please run your Jupyter Notebook first.")

# 3. UI Header
st.title("📊 User Behavior Clustering")
st.write("Enter user metrics below to identify their behavior group.")

# 4. User Inputs (Form banate hain taaki UI clean dikhe)
with st.form("user_input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        likes = st.number_input("Likes/Reactions", min_value=0)
        comments = st.number_input("Comments", min_value=0)
        shares = st.number_input("Shares/Retweets", min_value=0)
    
    with col2:
        followers = st.number_input("User Followers", min_value=0)
        engagement = st.number_input("User Engagement Score", min_value=0.0)
        interactions = st.number_input("Total User Interactions", min_value=0)

    submit = st.form_submit_button("Analyze User")

# 5. Prediction Logic
if submit:
    # Input data ko array mein convert karein
    input_data = np.array([[likes, comments, shares, followers, engagement, interactions]])
    
    # 🚨 Important: Scaling apply karein (wahi scaler use karke jo notebook mein tha)
    scaled_input = scaler.transform(input_data)
    
    # Prediction karein
    cluster = model.predict(scaled_input)[0]
    
    # Results dikhayein
    st.markdown("---")
    st.subheader(f"Result: User belongs to **Cluster {cluster}**")

    # Cluster Analysis (Profiling)
    if cluster == 0:
        st.success("👤 **Cluster 0: Passive User** - Low interaction, mostly viewing content.")
    elif cluster == 1:
        st.warning("👤 **Cluster 1: Moderate User** - Decent followers and occasional engagement.")
    else:
        st.info("🔥 **Cluster 2: Highly Active/Influencer** - High likes, shares, and massive interactions.")