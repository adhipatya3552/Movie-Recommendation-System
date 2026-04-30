import pickle
import pandas as pd

df = pd.read_csv("data/final_dataset.csv")

# Load models
collab_sim = pickle.load(open("models/collaborative.pkl", "rb"))
content_sim, movies = pickle.load(open("models/content.pkl", "rb"))

pivot = df.pivot_table(index='user_id', columns='title', values='rating').fillna(0)

# Collaborative
def get_collab_recommendations(movie):
    if movie not in pivot.columns:
        return []

    idx = list(pivot.columns).index(movie)
    scores = list(enumerate(collab_sim[idx]))

    movies_list = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    return [pivot.columns[i[0]] for i in movies_list]

# Content-based
def get_content_recommendations(movie):
    if movie not in movies['title'].values:
        return []

    idx = movies[movies['title'] == movie].index[0]
    scores = list(enumerate(content_sim[idx]))

    movies_list = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    return [movies.iloc[i[0]].title for i in movies_list]