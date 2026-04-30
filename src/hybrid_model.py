from src.collaborative_filtering import recommend_movies
from src.content_based import recommend_by_content

def hybrid_recommendation(movie_title):
    collab = recommend_movies(movie_title)
    content = recommend_by_content(movie_title)

    # Remove bad outputs
    collab = [m for m in collab if m != "Movie not found"]
    content = [m for m in content if m != "Movie not found"]

    combined = list(set(collab + content))
    return combined[:5]