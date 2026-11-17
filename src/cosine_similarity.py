import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load ratings
ratings = pd.read_csv(
    "dataset/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

# Create User-Movie Matrix
user_movie_matrix = ratings.pivot(
    index="user_id",
    columns="movie_id",
    values="rating"
).fillna(0)

# Calculate similarity between users
similarity_matrix = cosine_similarity(user_movie_matrix)

# Convert into DataFrame
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

# Display similarity
user_id = 1

print(f"\n===== USERS SIMILAR TO USER {user_id} =====")

# Get similarity scores for User 1
similar_users = similarity_df[user_id]

# Remove User 1 (self-similarity = 1)
similar_users = similar_users.drop(user_id)

# METHOD 1: Top-K Selection
top_k_users = similar_users.sort_values(
    ascending=False
).head(10)

print("\n===== TOP 10 SIMILAR USERS =====")
print(top_k_users)


# METHOD 2: Threshold Selection
threshold = 0.5

threshold_users = similar_users[
    similar_users >= threshold
].sort_values(ascending=False)

print(f"\n===== USERS WITH SIMILARITY >= {threshold} =====")
print(threshold_users)

print("\nTotal users selected using Top-K:", len(top_k_users))
print("Total users selected using Threshold:", len(threshold_users))
print(similar_users.iloc[1:6])
print("\nSimilarity Matrix Shape:")
print(similarity_matrix.shape)

print("\nSimilarity between User 1 and User 916:")
print(similarity_df.loc[1, 916])

print("\nSimilarity between User 916 and User 1:")
print(similarity_df.loc[916, 1])

print("\nSimilarity of User 1 with themselves:")
print(similarity_df.loc[1, 1])