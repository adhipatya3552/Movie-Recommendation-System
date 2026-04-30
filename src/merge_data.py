import pandas as pd

def merge_datasets():
    print("🔄 Loading datasets...")

    movies = pd.read_csv("data/movie.csv")
    ratings = pd.read_csv("data/rating.csv")
    tags = pd.read_csv("data/tag.csv")

    print("🔗 Merging datasets...")

    df = pd.merge(ratings, movies, on="movieId")
    df = pd.merge(df, tags, on=["userId", "movieId"], how="left")

    # Drop unnecessary columns safely
    df = df.drop(columns=[col for col in ["timestamp_x", "timestamp_y"] if col in df.columns])

    df.to_csv("data/final_dataset.csv", index=False)

    print("✅ Dataset merged successfully!")


if __name__ == "__main__":
    merge_datasets()