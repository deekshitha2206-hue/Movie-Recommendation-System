
import csv
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.models import User, Movie, Rating
from app.schemas.user import UserCreate, UserResponse
from app.schemas.rating import RatingCreate, RatingResponse
from app.services.recommendation_service import get_recommendations


app = FastAPI(
    title="Movie Recommendation System API",
    description="Backend API for the Movie Recommendation System",
    version="1.0.0"
)


# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MOVIES_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "ml-latest-small"
    / "movies.csv"
)


def load_movies():
    if not MOVIES_FILE.exists():
        raise FileNotFoundError(
            f"Movies dataset not found: {MOVIES_FILE}"
        )

    with open(
        MOVIES_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:
        return list(csv.DictReader(file))


movies = load_movies()


# Home API
@app.get("/")
def home():
    return {
        "status": "success",
        "message": "Movie Recommendation System API is running"
    }


# Get all movies
@app.get("/movies")
def get_movies(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100)
):
    return {
        "total_movies": len(movies),
        "skip": skip,
        "limit": limit,
        "movies": movies[skip:skip + limit]
    }


# Search movies
@app.get("/movies/search")
def search_movies(name: str = Query(min_length=1)):
    results = [
        movie for movie in movies
        if name.lower() in movie["title"].lower()
    ]

    return {
        "search_query": name,
        "total_results": len(results),
        "movies": results[:100]
    }


# Get movie by ID
@app.get("/movies/{movie_id}")
def get_movie(movie_id: int):
    for movie in movies:
        if int(movie["movieId"]) == movie_id:
            return movie

    raise HTTPException(
        status_code=404,
        detail="Movie not found"
    )


# Register user
@app.post(
    "/users",
    response_model=UserResponse,
    status_code=201
)
def create_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    existing_user = db.query(User).filter(
        (User.username == user_data.username) |
        (User.email == user_data.email)
    ).first()

    if existing_user:
        raise HTTPException(
            status_code=409,
            detail="Username or email already exists"
        )

    new_user = User(
        username=user_data.username,
        email=user_data.email
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user


# Add movie rating
@app.post(
    "/ratings",
    response_model=RatingResponse,
    status_code=201
)
def add_rating(
    rating_data: RatingCreate,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(
        User.id == rating_data.user_id
    ).first()

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found. Please register first."
        )

    movie = db.query(Movie).filter(
        Movie.id == rating_data.movie_id
    ).first()

    if not movie:
        raise HTTPException(
            status_code=404,
            detail="Movie not found"
        )

    new_rating = Rating(
        user_id=rating_data.user_id,
        movie_id=rating_data.movie_id,
        rating=rating_data.rating
    )

    db.add(new_rating)
    db.commit()
    db.refresh(new_rating)

    return {
        "message": "Rating added successfully",
        "user_id": rating_data.user_id,
        "movie_id": rating_data.movie_id,
        "rating": rating_data.rating
    }


# Get personalized movie recommendations
@app.get("/recommendations/{user_id}")
def recommend_movies(
    user_id: int,
    limit: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(
        User.id == user_id
    ).first()

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    recommendations = get_recommendations(
        db,
        user_id,
        limit
    )

    if not recommendations:
        raise HTTPException(
            status_code=404,
            detail="No recommendations found. Please add ratings first."
        )

    return {
        "user_id": user_id,
        "total_recommendations": len(recommendations),
        "recommendations": recommendations
    }
