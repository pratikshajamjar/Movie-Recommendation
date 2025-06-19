import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv('movies.csv')
df['overview'] = df['overview'].fillna('')

# TF-IDF and cosine similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Index mapping
df = df.reset_index()
indices = pd.Series(df.index, index=df['title'].str.lower())

# Recommendation function
def recommend(title):
    title = title.lower()
    if title not in indices:
        return []

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

# Streamlit UI
st.title("🎬 Movie Recommendation System")
movie_name = st.text_input("Enter a movie name:")

if st.button("Recommend"):
    if movie_name:
        recommendations = recommend(movie_name)
        if recommendations:
            st.subheader(f"Movies similar to *{movie_name.title()}*:")
            for rec in recommendations:
                st.write("👉", rec)
        else:
            st.warning("Movie not found. Please try another title.")
