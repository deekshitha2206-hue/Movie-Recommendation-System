
import csv
from pathlib import Path

from app.database import SessionLocal
from app.models.models import Movie

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MOVIES_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "ml-latest-small"
    / "movies.csv"
)


def seed_movies():
    db = SessionLocal()

    try:
        existing_count = db.query(Movie).count()

        if existing_count > 0:
            print(f"Database already contains {existing_count} movies.")
            print("Skipping import to avoid duplicate records.")
            return

        if not MOVIES_FILE.exists():
            raise FileNotFoundError(
                f"Movie dataset not found: {MOVIES_FILE}"
            )

        movie_records = []

        with open(
            MOVIES_FILE,
            "r",
            encoding="utf-8",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:
                movie = Movie(
                    id=int(row["movieId"]),
                    title=row["title"],
                    genres=row["genres"]
                )

                movie_records.append(movie)

        for start in range(0, len(movie_records), 1000):
            batch = movie_records[start:start + 1000]
            db.add_all(batch)

        db.commit()

        print("Movie data imported successfully!")
        print(f"Total movies imported: {len(movie_records)}")

    except Exception as error:
        db.rollback()
        print("Import failed:", error)

    finally:
        db.close()


if __name__ == "__main__":
    seed_movies()
