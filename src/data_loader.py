import pandas as pd

# Load movie ratings
ratings = pd.read_csv(
    "dataset/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

# Load movie names
movies = pd.read_csv(
    "dataset/u.item",
    sep="|",
    encoding="latin-1",
    usecols=[0, 1],
    names=["movie_id", "movie_title"]
)

# Display ratings
print("\n===== MOVIE RATINGS =====")
print(ratings.head())

# Display movies
print("\n===== MOVIE DETAILS =====")
print(movies.head())

# Dataset information
print("\n===== DATASET INFORMATION =====")
print("Total ratings:", len(ratings))
print("Total users:", ratings["user_id"].nunique())
print("Total movies:", ratings["movie_id"].nunique())