import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/cleaned_dataset.csv")

# -----------------------------
# FILTER DATA (VERY IMPORTANT)
# -----------------------------

# Keep movies with at least 100 ratings
movie_counts = df['title'].value_counts()
popular_movies = movie_counts[movie_counts > 100].index
df = df[df['title'].isin(popular_movies)]

# Keep active users
user_counts = df['userId'].value_counts()
active_users = user_counts[user_counts > 100].index
df = df[df['userId'].isin(active_users)]

print("Filtered dataset shape:", df.shape)

# -----------------------------
# CREATE MATRIX
# -----------------------------
user_movie_matrix = df.pivot_table(
    index='title',
    columns='userId',
    values='rating'
).fillna(0)

# Load model and matrix
model = pickle.load(open("models/knn_model.pkl", "rb"))
user_movie_matrix = pickle.load(open("models/matrix.pkl", "rb"))


def recommend_movies(movie_title, n_recommendations=5):
    if movie_title not in user_movie_matrix.index:
        return ["Movie not found"]

    movie_idx = user_movie_matrix.index.get_loc(movie_title)

    distances, indices = model.kneighbors(
        user_movie_matrix.iloc[movie_idx, :].values.reshape(1, -1),
        n_neighbors=n_recommendations + 1
    )

    recommendations = []

    for i in range(1, len(indices.flatten())):
        recommendations.append(user_movie_matrix.index[indices.flatten()[i]])

    return recommendations