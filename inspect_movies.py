
import csv

movies_file = "dataset/ml-latest-small/movies.csv"
ratings_file = "dataset/ml-latest-small/ratings.csv"

with open(movies_file, "r", encoding="utf-8") as file:
    movies = list(csv.DictReader(file))

with open(ratings_file, "r", encoding="utf-8") as file:
    ratings = list(csv.DictReader(file))

print("MOVIE DATASET")
print("Total movies:", len(movies))
print("First 5 movies:")

for movie in movies[:5]:
    print(movie)

print("\nRATINGS DATASET")
print("Total ratings:", len(ratings))
print("First 5 ratings:")

for rating in ratings[:5]:
    print(rating)
