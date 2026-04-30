# 🎬 Movie Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Engine-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas&logoColor=white)
![KNN](https://img.shields.io/badge/KNN-Collaborative%20Filtering-green?style=for-the-badge)

**A full movie recommendation engine that uses Content-Based Filtering, Collaborative Filtering, and a Hybrid approach — all in one project, with both a CLI and a Streamlit web interface.**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [How Each Recommendation Method Works](#-how-each-recommendation-method-works)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Getting Started](#-getting-started)
- [Running the Project](#-running-the-project)
- [Training the Model](#-training-the-model)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [API / Module Reference](#-api--module-reference)
- [Known Limitations](#-known-limitations)
- [Roadmap](#-roadmap)

---

## 🏥 Overview

This project is a **Movie Recommendation System** built entirely in Python. It takes a movie title as input and returns a list of movies the user would likely enjoy — based on different techniques from machine learning.

Three recommendation strategies are implemented:

- 🎯 **Content-Based Filtering** — recommends movies that are similar in genre to the one you picked
- 👥 **Collaborative Filtering** — recommends movies liked by other users who also liked the movie you picked (uses KNN)
- 🔀 **Hybrid Recommendation** — combines both methods above and removes duplicates to give the best of both worlds

You can run it either as a **command-line interface (CLI)** using `main.py`, or as a **Streamlit web app** using `app/app.py`.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Content-Based Filtering** | Uses movie genres + cosine similarity to find similar movies |
| 👥 **Collaborative Filtering** | Uses KNN model trained on a user-movie rating matrix |
| 🔀 **Hybrid Recommendations** | Combines both methods and gives top 5 unique results |
| 🖥️ **CLI Interface** | Menu-driven terminal app for quick testing |
| 🌐 **Streamlit Web App** | A simple, clean browser UI for non-technical users |
| 📊 **EDA Notebook** | Jupyter notebook for exploring the dataset visually |
| 🔗 **Data Merging** | Combines `movie.csv`, `rating.csv`, and `tag.csv` into one dataset |
| 🧹 **Data Cleaning** | Removes duplicates and handles missing values from the merged dataset |
| 💾 **Saved Models** | Trained KNN model and matrix saved using pickle for fast reuse |

---

## ⚙️ How Each Recommendation Method Works

### 1. 🎯 Content-Based Filtering (`src/content_based.py`)

This method looks at **what genres** a movie belongs to and finds other movies with similar genres.

Here's the step-by-step:
1. The cleaned dataset is loaded and duplicate movie titles are removed
2. The `genres` column (which looks like `"Action|Adventure|Sci-Fi"`) is passed through `CountVectorizer` — this converts genre strings into numerical vectors that a machine can compare
3. **Cosine similarity** is calculated between all movies using their genre vectors
4. When you give a movie title, it finds that movie's index, looks up its similarity scores against all other movies, sorts them from highest to lowest, and returns the top 5

> **In simple words:** If you pick "Toy Story" (a Children/Animation/Comedy movie), it will find other movies that also belong to those same genres.

---

### 2. 👥 Collaborative Filtering (`src/collaborative_filtering.py`)

This method doesn't care about genres at all. It looks at **how users rated movies** and finds patterns.

The idea is: if User A and User B both gave high ratings to similar movies, then a movie User A liked that User B hasn't seen yet is probably a good recommendation for User B.

Here's how it works:
1. The dataset is filtered to keep only **popular movies** (rated by more than 100 users) and **active users** (who rated more than 100 movies) — this removes noise
2. A **pivot table** (User-Movie matrix) is built where rows are movie titles, columns are user IDs, and the values are ratings (empty slots become 0)
3. A **K-Nearest Neighbors (KNN)** model is trained on this matrix using **cosine similarity** as the distance metric
4. This trained model and the matrix are saved as `.pkl` files using `pickle`
5. At recommendation time, the model is loaded, and when you give a movie title, it finds the K nearest neighbor movies in the rating space and returns them

> **In simple words:** It finds movies that were rated similarly by the same set of users — without knowing what genre those movies are.

---

### 3. 🔀 Hybrid Recommendation (`src/hybrid_model.py`)

This is the simplest of the three but also the most complete.

It:
1. Calls `recommend_movies()` (collaborative filtering) for a movie title
2. Calls `recommend_by_content()` (content-based) for the same title
3. Filters out any `"Movie not found"` errors from either
4. Combines both lists, removes duplicates using `set()`, and returns the top 5

> **In simple words:** It takes the best from both worlds. If one method misses something, the other one might catch it.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   USER INPUT                         │
│         (Movie Title from CLI or Web UI)             │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │      main.py / app.py   │
          │  (Entry Point / Router) │
          └────┬──────────┬─────────┘
               │          │
    ┌──────────▼──┐    ┌──▼──────────────┐
    │  Content-   │    │  Collaborative  │
    │  Based      │    │  Filtering      │
    │  Filtering  │    │  (KNN Model)    │
    │             │    │                 │
    │ CountVec +  │    │ User-Movie      │
    │ Cosine Sim  │    │ Matrix + KNN    │
    └──────┬──────┘    └────────┬────────┘
           │                    │
           └─────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │   Hybrid Model      │
          │  (Combines Both +   │
          │   Deduplication)    │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  Final Movie List   │
          │  (Top 5 Results)    │
          └─────────────────────┘
```

**Data Flow:**

```
movie.csv + rating.csv + tag.csv   (downloaded from Kaggle)
                 │
                 ▼
           merge_data.py  ──▶  final_dataset.csv
                                       │
                                       ▼
                              data_cleaning.py  ──▶  cleaned_dataset.csv
                                                              │
                          ┌───────────────────────────────────┤
                          ▼                     ▼             ▼
               content_based.py    collaborative_filtering.py  train_model.py
               (CountVec + Cos)    (Loads knn_model.pkl        (Trains & Saves
                                    and matrix.pkl)             KNN model)
                          │                     │
                          └──────────┬──────────┘
                                     ▼
                               hybrid_model.py
                                     │
                                     ▼
                             main.py / app/app.py
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.10+ | Core programming language |
| **Data Processing** | Pandas, NumPy | Loading, cleaning, pivoting the dataset |
| **Machine Learning** | Scikit-Learn | CountVectorizer, cosine_similarity, NearestNeighbors |
| **Model Saving** | Pickle | Saving and loading the trained KNN model and matrix |
| **Web UI** | Streamlit | Browser-based recommendation interface |
| **EDA** | Matplotlib, Seaborn | Data visualization in Jupyter notebook |
| **Notebook** | Jupyter | Exploratory data analysis |

---

## 📁 Project Structure

```
recommendation-system/
├── app/
│   └── app.py                     # Streamlit web application (UI)
│
├── data/
│   ├── movie.csv                  # Raw movies data downloaded from Kaggle
│   ├── rating.csv                 # Raw ratings data downloaded from Kaggle
│   ├── tag.csv                    # Raw tags data downloaded from Kaggle
│   ├── final_dataset.csv          # Merged dataset (generated by merge_data.py)
│   └── cleaned_dataset.csv        # Cleaned dataset (generated by data_cleaning.py)
│
├── models/
│   ├── knn_model.pkl              # Trained KNN model (generated by train_model.py)
│   └── matrix.pkl                 # User-Movie pivot matrix (generated by train_model.py)
│
├── notebooks/
│   └── eda.ipynb                  # Jupyter notebook for exploring the dataset
│
├── src/
│   ├── content_based.py           # Content-based filtering using genres + cosine similarity
│   ├── collaborative_filtering.py  # KNN-based collaborative filtering (loads saved model)
│   ├── hybrid_model.py            # Combines both methods and returns merged results
│   ├── train_model.py             # Trains the KNN model and saves it using pickle
│   └── utils.py                   # Alternate helper functions for both filtering methods
│
├── merge_data.py                  # Merges movie.csv, rating.csv, tag.csv into final_dataset.csv
├── data_cleaning.py               # Cleans final_dataset.csv and saves cleaned_dataset.csv
├── main.py                        # CLI entry point with menu-driven interface
└── requirements.txt               # Python dependencies
```

---

## 📊 Dataset

### Source

This project uses the **MovieLens 20M Dataset**, published by **[GroupLens Research](https://grouplens.org/)** at the University of Minnesota. It is one of the most well-known and widely used datasets in the recommendation systems community.

The dataset was taken from Kaggle:

> 🔗 **Kaggle Link:** [https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)

It contains:
- ~20 million ratings
- Across ~27,000 movies
- By ~138,000 users

The relevant data was combined and saved as `data/final_dataset.csv` for use in this project.

> ⚠️ **Note:** The dataset file is over 1.2 GB and is not included in this repository. Download it from the Kaggle link above and place the file at `data/final_dataset.csv` before running the project.

### Column Structure

| Column | Type | Description |
|--------|------|-------------|
| `userId` | Integer | Unique ID of the user who gave the rating |
| `movieId` | Integer | Unique ID of the movie |
| `rating` | Float | Rating given by the user (0.5 to 5.0 stars, in half-star steps) |
| `title` | String | Full title of the movie including release year, e.g., `"Jumanji (1995)"` |
| `genres` | String | Pipe-separated genre tags, e.g., `"Adventure|Children|Fantasy"` |
| `tag` | String | Optional user-submitted short description or keyword for the movie |

### Data Cleaning (done in `src/data_preprocessing.py`)

- Duplicate rows are dropped
- Missing values in the `tag` column are filled with `"No Tag"`
- The cleaned dataset is saved as `data/cleaned_dataset.csv`

### Filtering in Collaborative Filtering

To avoid noise from rarely-rated movies and inactive users:
- Only movies rated by **more than 100 users** are kept
- Only users who have rated **more than 100 movies** are kept

This makes the KNN model much more reliable.

### Citation

If you use this dataset in your own work, GroupLens requests the following citation:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 51:1–51:19. https://doi.org/10.1145/2827872

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or above
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/recommendation-system.git
cd recommendation-system
```

### 2. Create a Virtual Environment

It is recommended to create a virtual environment first so that all the libraries get installed in an isolated space and don't interfere with your system Python packages.

```bash
# Create virtual environment
python -m venv venv

# Activate it — on Windows
venv\Scripts\activate

# Activate it — on Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies

Once the virtual environment is active, install all required libraries:

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
```

### 4. Download the Dataset from Kaggle

Download the MovieLens 20M dataset from:

> 🔗 [https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)

After downloading, place the following three files inside the `data/` folder:
- `data/movie.csv`
- `data/rating.csv`
- `data/tag.csv`

### 5. Merge the Datasets

Run `merge_data.py` to combine all three raw files into a single dataset:

```bash
python merge_data.py
```

This merges `rating.csv` with `movie.csv` (on `movieId`) and then with `tag.csv` (on `userId` and `movieId`), drops unnecessary timestamp columns, and saves the result as `data/final_dataset.csv`.

### 6. Clean the Dataset

Once merging is done, run `data_cleaning.py` to clean the merged dataset:

```bash
python data_cleaning.py
```

This removes duplicate rows and fills any missing values in the `tag` column with `"No Tag"`, then saves the cleaned version as `data/cleaned_dataset.csv`.

### 7. Train the Collaborative Filtering Model

```bash
python src/train_model.py
```

This builds the user-movie matrix, trains the KNN model, and saves two files:
- `models/knn_model.pkl`
- `models/matrix.pkl`

> ⚠️ **This step is required** before using collaborative or hybrid recommendations. Without these files, the app will throw a file-not-found error.

---

## ▶️ Running the Project

### Option 1 — CLI (Terminal)

```bash
python main.py
```

You'll see a menu like this:

```
🎬 Movie Recommendation System
----------------------------------

Choose Recommendation Type:
1. Collaborative Filtering
2. Content-Based Filtering
3. Hybrid Recommendation
4. Exit

Enter your choice (1-4):
```

Type the number and enter a movie title (must match exactly as it appears in the dataset, including the year — e.g., `Toy Story (1995)`).

---

### Option 2 — Streamlit Web App

```bash
streamlit run app/app.py
```

The app opens in your browser at `http://localhost:8501`.

- It loads the cleaned dataset and shows a dropdown of all available movie titles
- Select a movie and click **Recommend**
- It uses the **content-based filtering** method and shows you the top 5 similar movies

> Note: The Streamlit app currently uses only content-based filtering. Collaborative and hybrid modes are accessible via `main.py`.

---

## 🧠 Training the Model

The KNN model is trained inside `src/train_model.py`. Here's what it does:

```python
# Filters noisy data
movie_counts = df['title'].value_counts()
popular_movies = movie_counts[movie_counts > 100].index
df = df[df['title'].isin(popular_movies)]

user_counts = df['userId'].value_counts()
active_users = user_counts[user_counts > 100].index
df = df[df['userId'].isin(active_users)]

# Creates user-movie pivot table
user_movie_matrix = df.pivot_table(
    index='title',
    columns='userId',
    values='rating'
).fillna(0)

# Trains KNN with cosine distance
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(user_movie_matrix)

# Saves model and matrix
pickle.dump(model, open("models/knn_model.pkl", "wb"))
pickle.dump(user_movie_matrix, open("models/matrix.pkl", "wb"))
```

The model uses **cosine similarity** because it focuses on the direction of rating patterns (not the magnitude), which works better when different users use different rating scales.

---

## 📓 Exploratory Data Analysis

The notebook at `notebooks/eda.ipynb` covers basic data exploration. Here's what it analyzes:

| Analysis | What It Shows |
|----------|--------------|
| `df.head()` | First few rows of the dataset |
| `df.info()` + `df.describe()` | Column types, count, and statistical summary |
| `df.isnull().sum()` | How many missing values exist per column |
| Unique users and movies count | Overall dataset size |
| Ratings distribution (histogram) | How ratings are spread — most users tend to rate 3.5–4.0 |
| Top 10 most rated movies (bar chart) | Most popular movies in the dataset |

To run the notebook:

```bash
jupyter notebook notebooks/eda.ipynb
```

---

## 📦 API / Module Reference

### `merge_data.py`

Loads `movie.csv`, `rating.csv`, and `tag.csv` from the `data/` folder, merges them into one combined dataset, drops unnecessary timestamp columns, and saves the result as `data/final_dataset.csv`.

```bash
python merge_data.py
```

---

### `data_cleaning.py`

Loads `final_dataset.csv`, removes duplicate rows, fills missing values in the `tag` column with `"No Tag"`, and saves the cleaned version as `data/cleaned_dataset.csv`.

```bash
python data_cleaning.py
```

---

### `src/content_based.py`

```python
recommend_by_content(movie_title, num_recommendations=5)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `movie_title` | `str` | Exact title of the movie as in the dataset |
| `num_recommendations` | `int` | Number of results to return (default: 5) |

**Returns:** A list of movie title strings.

---

### `src/collaborative_filtering.py`

```python
recommend_movies(movie_title, n_recommendations=5)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `movie_title` | `str` | Exact title of the movie as in the dataset |
| `n_recommendations` | `int` | Number of results to return (default: 5) |

**Returns:** A list of movie title strings. Returns `["Movie not found"]` if the title is not in the model's matrix.

> Requires `models/knn_model.pkl` and `models/matrix.pkl` to exist. Run `train_model.py` first.

---

### `src/hybrid_model.py`

```python
hybrid_recommendation(movie_title)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `movie_title` | `str` | Exact title of the movie |

**Returns:** A combined, deduplicated list of up to 5 movie recommendations.

---

### `src/train_model.py`

Trains the KNN model and saves the trained model + matrix to the `models/` folder.

```bash
python src/train_model.py
```

---

### `src/utils.py`

Contains two alternative helper functions used in earlier experiments:

| Function | What It Does |
|----------|-------------|
| `get_collab_recommendations(movie)` | Collaborative filtering using a pre-built similarity matrix loaded from `models/collaborative.pkl` |
| `get_content_recommendations(movie)` | Content-based using a pre-built similarity matrix loaded from `models/content.pkl` |

> Note: These utility functions expect a different set of pickle files compared to the main modules. They are kept for reference and experimentation purposes.

---

## ⚠️ Known Limitations

| Issue | Details |
|-------|---------|
| **Case-sensitive movie titles** | Movie names must match exactly, including year — e.g., `Toy Story (1995)` not `Toy Story` |
| **Cold start problem** | If a movie has fewer than 100 ratings, it gets filtered out and won't appear in collaborative results |
| **No real-time user data** | The model is static — it doesn't update as new ratings come in; retraining is needed |
| **Streamlit app is content-only** | The web interface only uses content-based filtering; collaborative and hybrid are CLI-only for now |
| **Large dataset size** | `final_dataset.csv` is over 1.2 GB; loading time may be slow on lower-end machines |
| **utils.py uses separate pkl files** | The helper functions in `utils.py` require `models/collaborative.pkl` and `models/content.pkl` which are separate from the main model files |

---

## 🗺️ Roadmap

- [x] Content-based filtering (genres + cosine similarity)
- [x] Collaborative filtering (KNN on user-movie matrix)
- [x] Hybrid recommendation (combined output)
- [x] CLI interface with menu
- [x] Streamlit web app
- [x] Dataset merging pipeline (`merge_data.py`)
- [x] Data cleaning pipeline (`data_cleaning.py`)
- [x] Model training and saving with pickle
- [x] EDA notebook
- [ ] Add collaborative and hybrid modes to the Streamlit app
- [ ] Add movie poster images in the web UI (via TMDB API)
- [ ] Matrix Factorization (SVD) for better accuracy
- [ ] User-based recommendation (not just item-based)
- [ ] Search bar with fuzzy matching for movie titles

---

<div align="center">

Built with ❤️ using Python, Scikit-Learn, and Streamlit.

</div>
