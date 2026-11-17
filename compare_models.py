
import csv
import numpy as np

from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

RATINGS_FILE = "dataset/ml-latest-small/ratings.csv"

K = 10
NEIGHBOR_COUNT = 20
RELEVANCE_THRESHOLD = 4.0

print("Loading MovieLens ratings...")

# 1. Load ratings
ratings_by_user = defaultdict(list)
all_user_ids = set()
all_movie_ids = set()

with open(RATINGS_FILE, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        user_id = int(row["userId"])
        movie_id = int(row["movieId"])
        rating = float(row["rating"])
        timestamp = int(row["timestamp"])

        ratings_by_user[user_id].append(
            (movie_id, rating, timestamp)
        )

        all_user_ids.add(user_id)
        all_movie_ids.add(movie_id)

user_ids = sorted(all_user_ids)
movie_ids = sorted(all_movie_ids)

user_map = {uid: i for i, uid in enumerate(user_ids)}
movie_map = {mid: i for i, mid in enumerate(movie_ids)}

# 2. Chronological train-test split
train_ratings = []
test_ratings = []

for user_id, user_ratings in ratings_by_user.items():

    user_ratings.sort(key=lambda x: x[2])

    if len(user_ratings) < 2:
        train_ratings.extend(
            (user_id, movie_id, rating)
            for movie_id, rating, timestamp in user_ratings
        )
        continue

    split_index = max(1, int(len(user_ratings) * 0.8))

    train_ratings.extend(
        (user_id, movie_id, rating)
        for movie_id, rating, timestamp
        in user_ratings[:split_index]
    )

    test_ratings.extend(
        (user_id, movie_id, rating)
        for movie_id, rating, timestamp
        in user_ratings[split_index:]
    )

print("Training ratings:", len(train_ratings))
print("Testing ratings:", len(test_ratings))

# 3. Create training user-item matrix
rows = [user_map[r[0]] for r in train_ratings]
cols = [movie_map[r[1]] for r in train_ratings]
values = [r[2] for r in train_ratings]

user_item_matrix = csr_matrix(
    (values, (rows, cols)),
    shape=(len(user_ids), len(movie_ids))
)

print("Training matrix:", user_item_matrix.shape)

# 4. Calculate user rating averages
user_means = {}

for user_id in user_ids:
    row = user_item_matrix.getrow(user_map[user_id])

    user_means[user_id] = (
        float(row.data.mean()) if row.nnz else 0.0
    )

global_mean = float(np.mean(values))

# 5. Prepare test data
relevant_by_user = defaultdict(set)

for user_id, movie_id, rating in test_ratings:

    if rating >= RELEVANCE_THRESHOLD:
        relevant_by_user[user_id].add(movie_id)

# 6. Train Item-Based model
print("\nTraining Item-Based model...")

item_matrix = user_item_matrix.T.tocsr()

item_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=min(NEIGHBOR_COUNT + 1, len(movie_ids)),
    n_jobs=1
)

item_model.fit(item_matrix)

item_distances, item_indices = item_model.kneighbors(item_matrix)

item_neighbors = {}

for movie_index in range(len(movie_ids)):

    neighbors = []

    for neighbor_index, distance in zip(
        item_indices[movie_index],
        item_distances[movie_index]
    ):

        if neighbor_index == movie_index:
            continue

        similarity = max(0.0, 1.0 - float(distance))

        if similarity > 0:
            neighbors.append((neighbor_index, similarity))

    item_neighbors[movie_index] = neighbors

print("Item-Based model trained!")

# 7. Train User-Based model
print("\nTraining User-Based model...")

user_similarity = cosine_similarity(user_item_matrix)

user_neighbors = {}

for user_index in range(len(user_ids)):

    similarities = user_similarity[user_index].copy()
    similarities[user_index] = 0

    neighbor_indices = np.argsort(similarities)[::-1]

    selected = [
        (index, similarities[index])
        for index in neighbor_indices
        if similarities[index] > 0
    ][:NEIGHBOR_COUNT]

    user_neighbors[user_index] = selected

print("User-Based model trained!")

# 8. Item-Based recommendations
def item_recommendations(user_id):

    user_index = user_map[user_id]
    user_row = user_item_matrix.getrow(user_index)

    rated_indices = set(user_row.indices)

    scores = np.zeros(len(movie_ids))
    weights = np.zeros(len(movie_ids))

    for movie_index, rating in zip(
        user_row.indices,
        user_row.data
    ):

        for neighbor_index, similarity in item_neighbors[movie_index]:

            if neighbor_index in rated_indices:
                continue

            scores[neighbor_index] += similarity * rating
            weights[neighbor_index] += similarity

    valid = weights > 0
    scores[valid] /= weights[valid]

    candidates = np.where(valid)[0]

    ranked = candidates[
        np.argsort(scores[candidates])[::-1]
    ][:K]

    return [movie_ids[index] for index in ranked]


# 9. User-Based recommendations
def user_recommendations(user_id):

    user_index = user_map[user_id]

    rated_indices = set(
        user_item_matrix.getrow(user_index).indices
    )

    neighbors = user_neighbors[user_index]

    if not neighbors:
        return []

    neighbor_indices = [item[0] for item in neighbors]
    similarities = np.array([item[1] for item in neighbors])

    neighbor_ratings = user_item_matrix[
        neighbor_indices
    ].toarray()

    weighted_sum = similarities @ neighbor_ratings

    weights = (
        (neighbor_ratings > 0) * similarities[:, None]
    ).sum(axis=0)

    valid = weights > 0

    scores = np.zeros(len(movie_ids))
    scores[valid] = weighted_sum[valid] / weights[valid]

    candidates = [
        index for index in np.where(valid)[0]
        if index not in rated_indices
    ]

    ranked = sorted(
        candidates,
        key=lambda index: scores[index],
        reverse=True
    )[:K]

    return [movie_ids[index] for index in ranked]


# 10. Predict Item-Based ratings
def predict_item_rating(user_id, movie_id):

    user_index = user_map[user_id]
    movie_index = movie_map[movie_id]

    user_row = user_item_matrix.getrow(user_index)

    neighbor_data = dict(item_neighbors[movie_index])

    numerator = 0.0
    denominator = 0.0

    for index, rating in zip(
        user_row.indices,
        user_row.data
    ):

        similarity = neighbor_data.get(index, 0.0)

        numerator += similarity * rating
        denominator += similarity

    if denominator == 0:
        return user_means[user_id] or global_mean

    return numerator / denominator


# 11. Predict User-Based ratings
def predict_user_rating(user_id, movie_id):

    user_index = user_map[user_id]
    movie_index = movie_map[movie_id]

    numerator = 0.0
    denominator = 0.0

    for neighbor_index, similarity in user_neighbors[user_index]:

        rating = user_item_matrix[
            neighbor_index, movie_index
        ]

        if rating > 0:
            numerator += similarity * float(rating)
            denominator += similarity

    if denominator == 0:
        return user_means[user_id] or global_mean

    return numerator / denominator


# 12. Evaluation
def evaluate(name, recommendation_function, prediction_function):

    print(f"\nEvaluating {name}...")

    precisions = []
    recalls = []
    all_recommended = set()

    # Generate recommendations for every user
    for user_id in user_ids:

        recommendations = recommendation_function(user_id)

        # Include all users in coverage
        all_recommended.update(recommendations)

        relevant = relevant_by_user[user_id]

        if not relevant:
            continue

        hits = len(set(recommendations) & relevant)

        precisions.append(hits / K)
        recalls.append(hits / len(relevant))

    # Evaluate rating predictions
    squared_errors = []
    absolute_errors = []

    for user_id, movie_id, actual_rating in test_ratings:

        predicted_rating = prediction_function(
            user_id,
            movie_id
        )

        predicted_rating = np.clip(
            predicted_rating, 0.5, 5.0
        )

        error = predicted_rating - actual_rating

        squared_errors.append(error ** 2)
        absolute_errors.append(abs(error))

    precision = np.mean(precisions) if precisions else 0.0
    recall = np.mean(recalls) if recalls else 0.0

    rmse = np.sqrt(np.mean(squared_errors))
    mae = np.mean(absolute_errors)

    # Catalogue coverage
    coverage = len(all_recommended) / len(movie_ids)

    return {
        "Precision@10": precision,
        "Recall@10": recall,
        "RMSE": rmse,
        "MAE": mae,
        "Coverage": coverage
    }


# 13. Compare both models
item_results = evaluate(
    "Item-Based Collaborative Filtering",
    item_recommendations,
    predict_item_rating
)

user_results = evaluate(
    "User-Based Collaborative Filtering",
    user_recommendations,
    predict_user_rating
)

# 14. Display comparison
print("\n" + "=" * 65)
print("             MODEL COMPARISON RESULTS")
print("=" * 65)

print(f"{'Metric':<18}{'Item-Based':>18}{'User-Based':>18}")

for metric in item_results:

    print(
        f"{metric:<18}"
        f"{item_results[metric]:>18.4f}"
        f"{user_results[metric]:>18.4f}"
    )

print("=" * 65)
print("\nMODEL COMPARISON COMPLETED!")
