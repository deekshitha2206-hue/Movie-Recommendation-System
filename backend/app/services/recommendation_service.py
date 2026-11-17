
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from app.models.models import Rating


# Dataset location
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "dataset" / "ml-latest-small"


# Load MovieLens datasets
with open(DATASET_PATH / "movies.csv", encoding="utf-8") as file:
    movies = list(csv.DictReader(file))

with open(DATASET_PATH / "ratings.csv", encoding="utf-8") as file:
    ratings = list(csv.DictReader(file))


# Create ID mappings
user_ids = sorted({int(r["userId"]) for r in ratings})
movie_ids = sorted({int(r["movieId"]) for r in ratings})

user_map = {uid: i for i, uid in enumerate(user_ids)}
movie_map = {mid: i for i, mid in enumerate(movie_ids)}


# Build rating matrix
rows, columns, values = [], [], []

for rating in ratings:
    rows.append(user_map[int(rating["userId"])])
    columns.append(movie_map[int(rating["movieId"])])
    values.append(float(rating["rating"]))

matrix = csr_matrix(
    (values, (rows, columns)),
    shape=(len(user_ids), len(movie_ids))
)

item_matrix = matrix.T.tocsr()


# Movie details
movie_details = {
    int(movie["movieId"]): {
        "title": movie["title"],
        "genres": movie["genres"]
    }
    for movie in movies
}


def cosine_similarity_rows(first, second):
    dot_product = first.multiply(second).sum()

    first_norm = np.sqrt(first.multiply(first).sum())
    second_norm = np.sqrt(second.multiply(second).sum())

    if first_norm == 0 or second_norm == 0:
        return 0.0

    return float(dot_product / (first_norm * second_norm))


def get_recommendations(db, user_id: int, limit: int = 10):

    user_ratings = (
        db.query(Rating)
        .filter(Rating.user_id == user_id)
        .all()
    )

    if not user_ratings:
        return []

    rated_movie_ids = {
        rating.movie_id for rating in user_ratings
    }

    liked_movies = [
        rating for rating in user_ratings
        if rating.rating >= 4
    ]

    if not liked_movies:
        return []

    score_totals = defaultdict(float)
    similarity_totals = defaultdict(float)

    for rating in liked_movies:

        movie_index = movie_map.get(rating.movie_id)

        if movie_index is None:
            continue

        source_movie = item_matrix[movie_index]

        similarities = []

        for candidate_index in range(len(movie_ids)):

            if candidate_index == movie_index:
                continue

            candidate_movie = item_matrix[candidate_index]

            similarity = cosine_similarity_rows(
                source_movie,
                candidate_movie
            )

            if similarity > 0:
                similarities.append((candidate_index, similarity))

        similarities.sort(
            key=lambda item: item[1],
            reverse=True
        )

        for similar_index, similarity in similarities[:10]:

            similar_movie_id = movie_ids[similar_index]

            if similar_movie_id in rated_movie_ids:
                continue

            score_totals[similar_movie_id] += (
                similarity * rating.rating
            )

            similarity_totals[similar_movie_id] += similarity

    recommendations = []

    for movie_id, total in score_totals.items():

        score = total / similarity_totals[movie_id]
        details = movie_details.get(movie_id)

        if details:
            recommendations.append({
                "movie_id": movie_id,
                "title": details["title"],
                "genres": details["genres"],
                "score": round(float(score), 2)
            })

    recommendations.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return recommendations[:limit]
