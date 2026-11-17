
import csv
from collections import defaultdict

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load datasets
with open("dataset/ml-latest-small/movies.csv", encoding="utf-8") as file:
    movies = list(csv.DictReader(file))

with open("dataset/ml-latest-small/ratings.csv", encoding="utf-8") as file:
    ratings = list(csv.DictReader(file))

# Create ID mappings
user_ids = sorted({int(r["userId"]) for r in ratings})
movie_ids = sorted({int(r["movieId"]) for r in ratings})

user_map = {uid: i for i, uid in enumerate(user_ids)}
movie_map = {mid: i for i, mid in enumerate(movie_ids)}

# Build rating matrix
rows, columns, values = [], [], []

for r in ratings:
    rows.append(user_map[int(r["userId"])])
    columns.append(movie_map[int(r["movieId"])])
    values.append(float(r["rating"]))

matrix = csr_matrix(
    (values, (rows, columns)),
    shape=(len(user_ids), len(movie_ids))
)

# Item-based similarity model
item_matrix = matrix.T.tocsr()

model = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=11,
    n_jobs=-1
)
model.fit(item_matrix)

movie_titles = {
    int(m["movieId"]): m["title"] for m in movies
}

# Select user
user_id = int(input("Enter User ID (1-610): "))

if user_id not in user_map:
    print("Invalid User ID.")
    raise SystemExit

user_index = user_map[user_id]
user_ratings = matrix.getrow(user_index)

# Movies already rated by user
rated_indices = set(user_ratings.indices)

# Consider movies rated 4 or above
liked_movies = [
    (index, rating)
    for index, rating in zip(
        user_ratings.indices, user_ratings.data
    )
    if rating >= 4
]

if not liked_movies:
    print("This user has no highly rated movies yet.")
    raise SystemExit

# Calculate recommendation scores
score_totals = defaultdict(float)
similarity_totals = defaultdict(float)

for movie_index, rating in liked_movies:
    distances, indices = model.kneighbors(
        item_matrix[movie_index],
        n_neighbors=11
    )

    for distance, similar_index in zip(distances[0], indices[0]):
        similarity = 1 - distance

        if similar_index == movie_index or similar_index in rated_indices:
            continue

        if similarity <= 0:
            continue

        score_totals[similar_index] += similarity * rating
        similarity_totals[similar_index] += similarity

# Rank recommendations
recommendations = []

for index, total in score_totals.items():
    score = total / similarity_totals[index]
    recommendations.append((index, score))

recommendations.sort(key=lambda x: x[1], reverse=True)

print(f"\nPersonalized recommendations for User {user_id}")
print("-" * 50)

for index, score in recommendations[:10]:
    movie_id = movie_ids[index]
    print(f"{movie_titles[movie_id]} | Score: {score:.2f}")
