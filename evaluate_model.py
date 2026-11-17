
import csv
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

print("Loading MovieLens dataset...")

# Load ratings
with open(
    "dataset/ml-latest-small/ratings.csv",
    encoding="utf-8"
) as file:
    ratings = list(csv.DictReader(file))

# Group ratings by user
user_ratings = {}

for row in ratings:
    uid = int(row["userId"])
    user_ratings.setdefault(uid, []).append(row)

# Chronological 80/20 train-test split
train_data = []
test_data = []

for uid, records in user_ratings.items():
    records.sort(key=lambda x: int(x["timestamp"]))

    if len(records) < 5:
        train_data.extend(records)
        continue

    split_index = int(len(records) * 0.8)

    train_data.extend(records[:split_index])
    test_data.extend(records[split_index:])

print("Training ratings:", len(train_data))
print("Testing ratings:", len(test_data))

# Create ID mappings
user_ids = sorted(user_ratings.keys())

movie_ids = sorted({
    int(row["movieId"]) for row in ratings
})

user_map = {
    uid: index for index, uid in enumerate(user_ids)
}

movie_map = {
    mid: index for index, mid in enumerate(movie_ids)
}

# Create training matrix
rows = []
columns = []
values = []

for row in train_data:
    rows.append(user_map[int(row["userId"])])
    columns.append(movie_map[int(row["movieId"])])
    values.append(float(row["rating"]))

train_matrix = csr_matrix(
    (values, (rows, columns)),
    shape=(len(user_ids), len(movie_ids))
)

print("\nTraining item-based recommendation model...")

# Item-user matrix
item_matrix = train_matrix.T.tocsr()

# Train model using one worker
model = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=21,
    n_jobs=1
)

model.fit(item_matrix)

# Calculate similarities only once
print("Calculating movie similarities...")

distances_all, indices_all = model.kneighbors(
    item_matrix,
    n_neighbors=21
)

print("Similarity calculation completed!")

# Group test ratings by user
test_by_user = {}

for row in test_data:
    uid = int(row["userId"])

    test_by_user.setdefault(uid, []).append(
        (int(row["movieId"]), float(row["rating"]))
    )

# Evaluation variables
K = 10

precision_scores = []
recall_scores = []

squared_errors = []
absolute_errors = []

covered_movies = set()

print("\nEvaluating recommendations...")

# Evaluate each user
for uid in user_ids:

    user_index = user_map[uid]
    user_vector = train_matrix.getrow(user_index)

    if uid not in test_by_user or user_vector.nnz == 0:
        continue

    # Already rated movies
    rated_indices = set(user_vector.indices)

    # Movies rated 4 or above
    liked_movies = [
        (index, rating)
        for index, rating in zip(
            user_vector.indices,
            user_vector.data
        )
        if rating >= 4
    ]

    scores = {}

    # Generate recommendations
    for movie_index, rating in liked_movies:

        distances = distances_all[movie_index]
        indices = indices_all[movie_index]

        for distance, similar_index in zip(
            distances, indices
        ):

            similarity = 1 - distance

            if similar_index in rated_indices:
                continue

            if similarity <= 0:
                continue

            scores[similar_index] = (
                scores.get(similar_index, 0)
                + similarity * rating
            )

    # Rank top 10 movies
    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:K]

    recommended_indices = [
        index for index, score in ranked
    ]

    covered_movies.update(recommended_indices)

    # Relevant test movies
    relevant = {
        movie_map[mid]
        for mid, rating in test_by_user[uid]
        if rating >= 4 and mid in movie_map
    }

    if relevant:
        hits = len(
            set(recommended_indices) & relevant
        )

        precision_scores.append(hits / K)
        recall_scores.append(hits / len(relevant))

    # Predict test ratings
    for movie_id, actual_rating in test_by_user[uid]:

        target_index = movie_map[movie_id]

        distances = distances_all[target_index]
        indices = indices_all[target_index]

        weighted_sum = 0
        similarity_sum = 0

        for distance, similar_index in zip(
            distances, indices
        ):

            similarity = 1 - distance

            if similar_index not in rated_indices:
                continue

            if similarity <= 0:
                continue

            position = np.where(
                user_vector.indices == similar_index
            )[0][0]

            user_rating = user_vector.data[position]

            weighted_sum += similarity * user_rating
            similarity_sum += similarity

        if similarity_sum > 0:
            predicted = weighted_sum / similarity_sum
        else:
            predicted = float(
                np.mean(user_vector.data)
            )

        predicted = max(0.5, min(5.0, predicted))

        squared_errors.append(
            (predicted - actual_rating) ** 2
        )

        absolute_errors.append(
            abs(predicted - actual_rating)
        )

# Display evaluation results
print("\n" + "=" * 45)
print("       MOVIE RECOMMENDATION EVALUATION")
print("=" * 45)

print("Training ratings:", len(train_data))
print("Testing ratings:", len(test_data))
print("Recommendation K:", K)

if precision_scores:
    print(
        "Precision@10:",
        round(np.mean(precision_scores), 4)
    )

    print(
        "Recall@10:",
        round(np.mean(recall_scores), 4)
    )

if squared_errors:
    rmse = np.sqrt(np.mean(squared_errors))
    mae = np.mean(absolute_errors)

    print("RMSE:", round(rmse, 4))
    print("MAE:", round(mae, 4))

coverage = (
    len(covered_movies) / len(movie_ids)
) * 100

print("Catalog coverage:", round(coverage, 2), "%")

print("=" * 45)
print("EVALUATION COMPLETED!")
