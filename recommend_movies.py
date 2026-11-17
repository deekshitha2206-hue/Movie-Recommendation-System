
import csv
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Dataset paths
movies_file = "dataset/ml-latest-small/movies.csv"
ratings_file = "dataset/ml-latest-small/ratings.csv"

# Load movies
with open(movies_file, "r", encoding="utf-8") as file:
    movies = list(csv.DictReader(file))

# Load ratings
with open(ratings_file, "r", encoding="utf-8") as file:
    ratings = list(csv.DictReader(file))

# Get unique IDs
user_ids = sorted({int(row["userId"]) for row in ratings})
movie_ids = sorted({int(row["movieId"]) for row in ratings})

# Create ID mappings
user_map = {uid: i for i, uid in enumerate(user_ids)}
movie_map = {mid: i for i, mid in enumerate(movie_ids)}

# Create user-item matrix
rows = []
columns = []
values = []

for rating in ratings:
    rows.append(user_map[int(rating["userId"])])
    columns.append(movie_map[int(rating["movieId"])])
    values.append(float(rating["rating"]))

rating_matrix = csr_matrix(
    (values, (rows, columns)),
    shape=(len(user_ids), len(movie_ids))
)

# Convert to item-user matrix
item_matrix = rating_matrix.T.tocsr()

# Train item similarity model
model = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=6,
    n_jobs=-1
)

model.fit(item_matrix)

# Movie lookup
movie_lookup = {
    int(movie["movieId"]): movie["title"]
    for movie in movies
}

print("MOVIE RECOMMENDATION SYSTEM READY!")

# Ask user for a movie
search_title = input("\nEnter a movie name: ").strip().lower()

matches = [
    (movie_id, movie_lookup[movie_id])
    for movie_id in movie_ids
    if search_title in movie_lookup[movie_id].lower()
]

if not matches:
    print("Movie not found. Try another movie name.")
else:
    print("\nMatching movies:")
    for movie_id, title in matches[:10]:
        print(f"{movie_id}: {title}")

    selected_id = matches[0][0]
    selected_index = movie_map[selected_id]

    # Find similar movies
    distances, indices = model.kneighbors(
        item_matrix[selected_index],
        n_neighbors=6
    )

    print("\nRecommendations for:", movie_lookup[selected_id])
    print("-" * 45)

    count = 0

    for distance, index in zip(distances[0], indices[0]):
        recommended_id = movie_ids[index]

        if recommended_id == selected_id:
            continue

        similarity = 1 - distance

        print(
            f"{movie_lookup[recommended_id]} "
            f"| Similarity: {similarity:.3f}"
        )

        count += 1
        if count == 5:
            break
