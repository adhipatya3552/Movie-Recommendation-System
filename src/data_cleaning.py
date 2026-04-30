import pandas as pd

def clean_dataset():
    print("🧹 Cleaning dataset...")

    df = pd.read_csv("data/final_dataset.csv")

    # Remove duplicates
    df = df.drop_duplicates()

    # Fill missing tags
    if "tag" in df.columns:
        df["tag"] = df["tag"].fillna("No Tag")

    df.to_csv("data/cleaned_dataset.csv", index=False)

    print("✅ Data cleaning completed!")


if __name__ == "__main__":
    clean_dataset()