
import pandas as pd

# Load MovieLens datasets
movies = pd.read_csv("dataset/ml-latest-small/movies.csv")
ratings = pd.read_csv("dataset/ml-latest-small/ratings.csv")

# Explore movies dataset
print("\n========== MOVIES DATASET ==========")

print("\nFirst 5 Movies:")
print(movies.head())

print("\nMovies Shape:")
print(movies.shape)

print("\nMovies Columns:")
print(movies.columns)

# Explore ratings dataset
print("\n========== RATINGS DATASET ==========")

print("\nFirst 5 Ratings:")
print(ratings.head())

print("\nRatings Shape:")
print(ratings.shape)

print("\nRatings Columns:")
print(ratings.columns)

# Check missing values
print("\n========== MISSING VALUES ==========")

print("\nMovies:")
print(movies.isnull().sum())

print("\nRatings:")
print(ratings.isnull().sum())

# Rating distribution
print("\n========== RATING DISTRIBUTION ==========")
print(ratings["rating"].value_counts().sort_index())
