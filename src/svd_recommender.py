import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds


# STEP 1: Load the dataset

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

R = user_movie_matrix.to_numpy(dtype=float)

print("Original Matrix Shape:", R.shape)


# STEP 3: Perform Singular Value Decomposition

k = 20

U, sigma, VT = svds(R, k=k)


# STEP 4: Sort Singular Values in Descending Order

idx = np.argsort(sigma)[::-1]

U = U[:, idx]
sigma = sigma[idx]
VT = VT[idx, :]

Sigma = np.diag(sigma)


# STEP 5: Reconstruct the Predicted Rating Matrix

predicted_ratings = U @ Sigma @ VT

print("U Shape:", U.shape)
print("Sigma Shape:", Sigma.shape)
print("VT Shape:", VT.shape)
print("Predicted Matrix Shape:", predicted_ratings.shape)


# STEP 6: Generate Movie Recommendations

def recommend_movies(user_id, top_n=5):

    if user_id not in user_movie_matrix.index:
        print("Invalid User ID")
        return

    # Get the user's row position
    user_index = user_movie_matrix.index.get_loc(user_id)

    # Get predicted ratings for that user
    user_predictions = predicted_ratings[user_index]

    # Convert predictions into a Series
    predictions = pd.Series(
        user_predictions,
        index=user_movie_matrix.columns
    )

    # Identify movies already rated by the user
    already_rated = user_movie_matrix.loc[user_id]
    already_rated = already_rated[already_rated > 0].index

    # Remove already-rated movies
    predictions = predictions.drop(already_rated)

    # Select top-rated predicted movies
    top_movies = predictions.sort_values(
        ascending=False
    ).head(top_n)

    # Display movie titles
    print(f"\nTop {top_n} Movie Recommendations for User {user_id}:\n")

    for movie_id, predicted_rating in top_movies.items():

        movie_title = movies.loc[
            movies["movie_id"] == movie_id,
            "movie_title"
        ].iloc[0]

        print(f"{movie_title} --> Predicted Rating: {predicted_rating:.2f}")


# STEP 7: User Input

user_id = int(input("\nEnter User ID (1-943): "))

recommend_movies(user_id)