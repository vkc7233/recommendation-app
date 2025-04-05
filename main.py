import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🔍 SHL Assessment Recommendation Engine By Vineet Kumar Chaturvedi")

# File uploader
uploaded_file = st.file_uploader("Upload SHL Product Catalogue (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Combine relevant fields
    df['combined_features'] = df['job_roles'].fillna('') + " " + \
                              df['skills'].fillna('') + " " + \
                              df['level'].fillna('')

    # Vectorize catalogue features
    vectorizer = TfidfVectorizer()
    feature_matrix = vectorizer.fit_transform(df['combined_features'])

    # User input
    user_input = st.text_input("Enter role, skills, level (e.g., 'Data Analyst Python Entry')")

    if user_input:
        user_vector = vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, feature_matrix).flatten()

        top_indices = similarities.argsort()[-5:][::-1]
        st.subheader("📌 Top Recommended Assessments:")

        for i, idx in enumerate(top_indices, start=1):
            row = df.iloc[idx]
            st.markdown(f"""
            **{i}. {row['title']}**  
            🧑‍💼 Job Roles: {row['job_roles']}  
            🛠️ Skills: {row['skills']}  
            🏷️ Level: {row['level']}  
            ---
            """)

else:
    st.info("Please upload a CSV file with columns like: title, job_roles, skills, level.")
