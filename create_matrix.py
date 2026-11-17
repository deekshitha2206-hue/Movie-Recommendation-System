
import csv
import numpy as np
from scipy.sparse import csr_matrix

ratings_file = "dataset/ml-latest-small/ratings.csv"

# Read ratings
with open(ratings_file, "r", encoding="utf-8") as file:
    ratings = list(csv.DictReader(file))

# Get unique user and movie IDs
user_ids = sorted({int(row["userId"]) for row in ratings})
movie_ids = sorted({int(row["movieId"]) for row in ratings})

# Map IDs to matrix positions
user_map = {user_id: index for index, user_id in enumerate(user_ids)}
movie_map = {movie_id: index for index, movie_id in enumerate(movie_ids)}

# Prepare matrix data
row_indices = []
column_indices = []
rating_values = []

for row in ratings:
    row_indices.append(user_map[int(row["userId"])])
    column_indices.append(movie_map[int(row["movieId"])])
    rating_values.append(float(row["rating"]))

# Create sparse user-item matrix
rating_matrix = csr_matrix(
    (rating_values, (row_indices, column_indices)),
    shape=(len(user_ids), len(movie_ids))
)

print("USER-ITEM MATRIX CREATED SUCCESSFULLY!")
print("Matrix shape:", rating_matrix.shape)
print("Total ratings:", rating_matrix.nnz)
print("Total users:", len(user_ids))
print("Total movies:", len(movie_ids))

total_cells = len(user_ids) * len(movie_ids)
density = (rating_matrix.nnz / total_cells) * 100

print("Matrix density:", round(density, 2), "%")

print("\nFirst 5 users and first 5 movies:")
print(rating_matrix[:5, :5].toarray())
