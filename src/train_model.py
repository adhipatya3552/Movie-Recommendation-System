import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors

# Load dataset
df = pd.read_csv("data/cleaned_dataset.csv")

# Filter data
movie_counts = df['title'].value_counts()
popular_movies = movie_counts[movie_counts > 100].index
df = df[df['title'].isin(popular_movies)]

user_counts = df['userId'].value_counts()
active_users = user_counts[user_counts > 100].index
df = df[df['userId'].isin(active_users)]

# Create matrix
user_movie_matrix = df.pivot_table(
    index='title',
    columns='userId',
    values='rating'
).fillna(0)

# Train model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(user_movie_matrix)

# ✅ SAVE MODEL
pickle.dump(model, open("models/knn_model.pkl", "wb"))

# ✅ SAVE MATRIX (VERY IMPORTANT)
pickle.dump(user_movie_matrix, open("models/matrix.pkl", "wb"))

print("✅ Model and matrix saved successfully!")