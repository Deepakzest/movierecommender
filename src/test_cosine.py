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

print("User-Movie Matrix Shape:", user_movie_matrix.shape)


# STEP 3: Calculate Cosine Similarity

similarity_matrix = cosine_similarity(user_movie_matrix)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

print("Similarity Matrix Shape:", similarity_df.shape)


# STEP 4: Generate Recommendations

def recommend_movies(user_id, top_n=5):

    if user_id not in user_movie_matrix.index:
        print("Invalid User ID")
        return

    # Find the top 10 similar users
    similar_users = (
        similarity_df[user_id]
        .drop(user_id) #user 1 is 100% similar to user 1
        .sort_values(ascending=False)
        .head(10)
    )

    # Store predicted ratings
    predictions = {}

    # Find movies not rated by the target user
    unrated_movies = user_movie_matrix.loc[user_id]
    unrated_movies = unrated_movies[unrated_movies == 0].index #We only recommend movies the user hasn't rated.

    # Predict rating for each unrated movie
    for movie_id in unrated_movies:

        weighted_sum = 0
        similarity_sum = 0

        for similar_user_id, similarity_score in similar_users.items():

            rating = user_movie_matrix.loc[
                similar_user_id, movie_id
            ]

            if rating > 0:

                weighted_sum += similarity_score * rating
                similarity_sum += similarity_score

        # Calculate weighted average
        if similarity_sum > 0:

            predicted_rating = weighted_sum / similarity_sum

            predictions[movie_id] = predicted_rating

    # Sort predicted ratings
    top_movies = sorted(
        predictions.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    # Display recommendations
    print(f"\nTop {top_n} Recommendations for User {user_id}:\n")

    for movie_id, predicted_rating in top_movies:

        movie_title = movies.loc[
            movies["movie_id"] == movie_id,
            "movie_title"
        ].iloc[0]

        print(
            f"{movie_title} --> Predicted Rating: {predicted_rating:.2f}"
        )


# STEP 5: User Input

user_id = int(input("\nEnter User ID (1-943): "))

recommend_movies(user_id)