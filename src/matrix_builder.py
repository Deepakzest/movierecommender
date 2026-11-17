import pandas as pd

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
)

# Replace missing ratings with 0
user_movie_matrix = user_movie_matrix.fillna(0)

# Display matrix information
print("===== USER-MOVIE MATRIX =====")

print("Matrix Shape:", user_movie_matrix.shape)

print("\nFirst 5 Users and First 5 Movies:")
print(user_movie_matrix.iloc[:5, :5])

# Convert to NumPy matrix
matrix = user_movie_matrix.to_numpy()

print("\nNumPy Matrix Shape:", matrix.shape)