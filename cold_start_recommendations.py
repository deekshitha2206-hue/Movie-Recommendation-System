
import csv

MOVIES_FILE = "dataset/ml-latest-small/movies.csv"
RATINGS_FILE = "dataset/ml-latest-small/ratings.csv"

print("Loading MovieLens dataset...")

# Load movie titles
movies = {}

with open(MOVIES_FILE, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        movies[int(row["movieId"])] = row["title"]

# Collect rating count and rating total
rating_stats = {}

with open(RATINGS_FILE, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        movie_id = int(row["movieId"])
        rating = float(row["rating"])

        if movie_id not in rating_stats:
            rating_stats[movie_id] = {
                "total": 0,
                "count": 0
            }

        rating_stats[movie_id]["total"] += rating
        rating_stats[movie_id]["count"] += 1

# Calculate global average rating
total_ratings = sum(
    stats["count"] for stats in rating_stats.values()
)

global_average = (
    sum(stats["total"] for stats in rating_stats.values())
    / total_ratings
)

# Calculate popularity-weighted scores
minimum_ratings = 20
popular_movies = []

for movie_id, stats in rating_stats.items():
    count = stats["count"]
    average = stats["total"] / count

    if count >= minimum_ratings:
        weighted_score = (
            (count / (count + minimum_ratings)) * average
            + (minimum_ratings / (count + minimum_ratings))
            * global_average
        )

        popular_movies.append((
            movie_id,
            weighted_score,
            count
        ))

# Sort by weighted score
popular_movies.sort(key=lambda x: x[1], reverse=True)


def recommend_popular_movies(top_n=10):
    print("\nPopular Movie Recommendations")
    print("-" * 55)

    for movie_id, score, count in popular_movies[:top_n]:
        print(
            f"{movies[movie_id]} | "
            f"Score: {score:.2f} | Ratings: {count}"
        )


def recommend_for_new_user():
    print("\nNew user detected!")
    print("Showing popular movies because no rating history exists.")
    recommend_popular_movies()


def handle_new_movie(movie_id):
    if movie_id not in movies:
        print("\nMovie ID not found in the dataset.")
        return

    if movie_id not in rating_stats:
        print(f"\n{movies[movie_id]} has no ratings yet.")
        print("Using popular movies as a fallback:")
        recommend_popular_movies()
    else:
        print(f"\n{movies[movie_id]} already has rating history.")
        print("It can be considered by the recommendation model.")


# Test 1: New user
recommend_for_new_user()

# Test 2: New or unrated movie
handle_new_movie(9999999)
