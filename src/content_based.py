import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/cleaned_dataset.csv")

# Use genres as features
df = df.drop_duplicates(subset="title")

cv = CountVectorizer(stop_words='english')
matrix = cv.fit_transform(df['genres'])

similarity = cosine_similarity(matrix)


def recommend_by_content(movie_title, num_recommendations=5):
    idx = df[df['title'] == movie_title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    movie_indices = [i[0] for i in scores]
    return df['title'].iloc[movie_indices].tolist()