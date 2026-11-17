
import csv
from pathlib import Path
from collections import defaultdict

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from app.models.models import Rating, Movie


# Project dataset location
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


# Train Item-Based Collaborative Filtering model
item_matrix = matrix.T.tocsr()

model = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=11,
    n_jobs=-1
)

model.fit(item_matrix)


# Movie details
movie_details = {
    int(movie["movieId"]): {
        "title": movie["title"],
        "genres": movie["genres"]
    }
    for movie in movies
}


def get_recommendations(db, user_id: int, limit: int = 10):

    # Get ratings submitted by this application user
    user_ratings = (
        db.query(Rating)
        .filter(Rating.user_id == user_id)
        .all()
    )

    if not user_ratings:
        return []

    # Movies already rated by this user
    rated_movie_ids = {
        rating.movie_id for rating in user_ratings
    }

    # Consider movies rated 4 or above
    liked_movies = [
        rating
        for rating in user_ratings
        if rating.rating >= 4
    ]

    if not liked_movies:
        return []

    score_totals = defaultdict(float)
    similarity_totals = defaultdict(float)

    # Find similar movies
    for rating in liked_movies:

        movie_index = movie_map.get(rating.movie_id)

        if movie_index is None:
            continue

        distances, indices = model.kneighbors(
            item_matrix[movie_index],
            n_neighbors=11
        )

        for distance, similar_index in zip(
            distances[0], indices[0]
        ):

            similar_movie_id = movie_ids[similar_index]
            similarity = 1 - distance

            if similar_movie_id in rated_movie_ids:
                continue

            if similarity <= 0:
                continue

            score_totals[similar_movie_id] += (
                similarity * rating.rating
            )

            similarity_totals[similar_movie_id] += similarity

    # Rank recommendations
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
