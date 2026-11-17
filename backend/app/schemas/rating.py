from pydantic import BaseModel, Field


class RatingCreate(BaseModel):
    user_id: int = Field(gt=0)
    movie_id: int = Field(gt=0)
    rating: float = Field(ge=0.5, le=5.0)


class RatingResponse(BaseModel):
    message: str
    user_id: int
    movie_id: int
    rating: float