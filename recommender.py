import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and prepare data
df = pd.read_csv("data.csv")
df['combined_features'] = df['job_roles'].fillna('') + " " + df['skills'].fillna('') + " " + df['level'].fillna('')
vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(df['combined_features'])

def get_recommendations(query: str):
    query_vec = vectorizer.transform([query])
    sim_scores = cosine_similarity(query_vec, matrix).flatten()
    top_indices = sim_scores.argsort()[-10:][::-1]
    
    results = []
    for i in top_indices:
        row = df.iloc[i]
        results.append({
            "url": row.get("url", ""),
            "adaptive_support": row.get("adaptive_support", "No"),
            "description": row.get("description", ""),
            "duration": int(row.get("duration", 0)),
            "remote_support": row.get("remote_support", "No"),
            "test_type": row.get("test_type", "").split(",")  # should be list of strings
        })
    return results
