import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# STEP 1: Load Dataset
ratings = pd.read_csv(
    "dataset/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

movies = pd.read_csv(
    "dataset/u.item",
    sep="|",
    encoding="latin-1",
    usecols=[0, 1],
    names=["movie_id", "movie_title"]
)


# STEP 2: Create User-Movie Matrix

user_movie_matrix = ratings.pivot(
    index="user_id",
    columns="movie_id",
    values="rating"
).fillna(0)

print("User-Movie Matrix Shape:", user_movie_matrix)
similarity_matrix = cosine_similarity(user_movie_matrix)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

print("Similarity Matrix Shape:", similarity_df)