import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from src.content_based import recommend_by_content

# ---------- CONFIG ----------
st.set_page_config(page_title="Movie Recommender", layout="wide")

# ---------- LOAD DATA ----------
df = pd.read_csv("data/cleaned_dataset.csv")
movie_list = df['title'].unique()

# ---------- UI HEADER ----------
st.markdown(
    "<h1 style='color:red;'>🎬 Movie Recommendation System</h1>",
    unsafe_allow_html=True
)

st.markdown("### Discover movies you’ll love 🍿")

# ---------- SELECT MOVIE ----------
selected_movie = st.selectbox("Choose a movie:", movie_list)

# ---------- RECOMMEND ----------
if st.button("Recommend"):
    recommendations = recommend_by_content(selected_movie)

    st.markdown("## 🔥 Top Picks For You")

    cols = st.columns(5)

    for i, movie in enumerate(recommendations):
        st.write(movie)