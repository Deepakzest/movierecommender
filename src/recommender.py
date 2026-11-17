import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load ratings
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

# Create User-Movie Matrix
user_movie_matrix = ratings.pivot(
    index="user_id",
    columns="movie_id",
    values="rating"
).fillna(0)

# Calculate similarity
similarity_matrix = cosine_similarity(user_movie_matrix)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)


def recommend_movies(user_id, top_n=5):

    # Find similar users
    similar_users = similarity_df[user_id].drop(user_id)

    similar_users = similar_users.sort_values(
        ascending=False
    ).head(10)

    # Calculate average rating of target user
    user_mean = user_movie_matrix.loc[user_id][
        user_movie_matrix.loc[user_id] > 0
    ].mean()

    watched_movies = user_movie_matrix.loc[user_id]

    recommendations = {}

    for movie_id in user_movie_matrix.columns:

        if watched_movies[movie_id] > 0:
            continue

        weighted_sum = 0
        similarity_sum = 0

        for other_user, similarity in similar_users.items():

            rating = user_movie_matrix.loc[other_user, movie_id]

            if rating > 0:

                # Calculate other user's average rating
                other_user_ratings = user_movie_matrix.loc[other_user]

                other_user_mean = other_user_ratings[
                    other_user_ratings > 0
                ].mean()

                # Calculate rating deviation
                deviation = rating - other_user_mean

                weighted_sum += similarity * deviation

                similarity_sum += abs(similarity)

        if similarity_sum > 0:

            predicted_rating = user_mean + (
                weighted_sum / similarity_sum
            )

            # Keep prediction within 1-5 range
            predicted_rating = max(1, min(5, predicted_rating))

            recommendations[movie_id] = predicted_rating

    recommendations = sorted(
        recommendations.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\n===== RECOMMENDATIONS FOR USER {user_id} =====")

    for movie_id, predicted_rating in recommendations[:top_n]:

        movie_name = movies.loc[
            movies["movie_id"] == movie_id,
            "movie_title"
        ].values[0]

        print(
            f"{movie_name} -> Predicted Rating: "
            f"{predicted_rating:.2f}"
        )

# Test
recommend_movies(1)