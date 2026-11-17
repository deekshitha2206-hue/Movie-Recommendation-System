
import csv
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Dataset paths
MOVIES_FILE = "dataset/ml-latest-small/movies.csv"
RATINGS_FILE = "dataset/ml-latest-small/ratings.csv"

print("Loading MovieLens dataset...")

# Read movies
movies = {}

with open(MOVIES_FILE, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        movies[int(row["movieId"])] = row["title"]

# Read ratings
ratings = []

with open(RATINGS_FILE, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        ratings.append((
            int(row["userId"]),
            int(row["movieId"]),
            float(row["rating"])
        ))

# Create user and movie mappings
user_ids = sorted(set(r[0] for r in ratings))
movie_ids = sorted(set(r[1] for r in ratings))

user_map = {user_id: index for index, user_id in enumerate(user_ids)}
movie_map = {movie_id: index for index, movie_id in enumerate(movie_ids)}

# Create user-item sparse matrix
rows = [user_map[r[0]] for r in ratings]
cols = [movie_map[r[1]] for r in ratings]
values = [r[2] for r in ratings]

user_item_matrix = csr_matrix(
    (values, (rows, cols)),
    shape=(len(user_ids), len(movie_ids))
)

print("User-item matrix created:", user_item_matrix.shape)

# Calculate user-user cosine similarity
print("\nCalculating user similarities...")

similarity_matrix = cosine_similarity(user_item_matrix)

print("User similarity calculation completed!")

# Generate recommendations
def recommend_for_user(target_user_id, top_n=10, neighbor_count=10):

    if target_user_id not in user_map:
        print("User not found in the dataset.")
        return

    target_index = user_map[target_user_id]

    # Get similarity scores with other users
    similarities = similarity_matrix[target_index].copy()

    # Exclude the target user
    similarities[target_index] = 0

    # Select most similar users
    neighbor_indices = np.argsort(similarities)[::-1]
    neighbor_indices = [
        index for index in neighbor_indices
        if similarities[index] > 0
    ][:neighbor_count]

    if not neighbor_indices:
        print("No similar users found.")
        return

    # Get movies already rated by target user
    user_ratings = user_item_matrix[target_index].toarray().flatten()
    rated_movies = set(np.where(user_ratings > 0)[0])

    # Calculate weighted scores for unseen movies
    scores = np.zeros(len(movie_ids))
    weights = np.zeros(len(movie_ids))

    for neighbor_index in neighbor_indices:
        similarity = similarities[neighbor_index]
        neighbor_ratings = user_item_matrix[neighbor_index].toarray().flatten()

        for movie_index in np.where(neighbor_ratings > 0)[0]:
            if movie_index not in rated_movies:
                scores[movie_index] += (
                    similarity * neighbor_ratings[movie_index]
                )
                weights[movie_index] += similarity

    # Normalize scores
    valid = weights > 0
    scores[valid] = scores[valid] / weights[valid]

    # Rank unseen movies
    candidate_indices = np.where(valid)[0]
    ranked_indices = candidate_indices[
        np.argsort(scores[candidate_indices])[::-1]
    ][:top_n]

    print(f"\nRecommendations for User ID: {target_user_id}")
    print("-" * 55)

    if len(ranked_indices) == 0:
        print("No recommendations available.")
        return

    for index in ranked_indices:
        movie_id = movie_ids[index]
        title = movies.get(movie_id, "Unknown movie")
        print(f"{title} | Predicted score: {scores[index]:.2f}")


# Run recommendation for User 1
recommend_for_user(target_user_id=1)
