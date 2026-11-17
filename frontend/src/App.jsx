

import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [movies, setMovies] = useState([]);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [userId, setUserId] = useState("1");
  const [recommendations, setRecommendations] = useState([]);
  const [recommendationLoading, setRecommendationLoading] = useState(false);
  const [recommendationError, setRecommendationError] = useState("");

  useEffect(() => {
    fetchMovies();
  }, []);

  async function fetchMovies() {
    try {
      setLoading(true);
      setError("");

      const response = await fetch(`${API_URL}/movies?skip=0&limit=100`);

      if (!response.ok) {
        throw new Error("Unable to load movies");
      }

      const data = await response.json();
      setMovies(data.movies);
    } catch (err) {
      setError("Backend connection failed. Please start the FastAPI server.");
    } finally {
      setLoading(false);
    }
  }

  async function searchMovies(event) {
    event.preventDefault();

    if (!search.trim()) {
      fetchMovies();
      return;
    }

    try {
      setLoading(true);
      setError("");

      const response = await fetch(
        `${API_URL}/movies/search?name=${encodeURIComponent(search)}`
      );

      if (!response.ok) {
        throw new Error("Search failed");
      }

      const data = await response.json();
      setMovies(data.movies);
    } catch (err) {
      setError("Unable to search movies. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  async function fetchRecommendations(event) {
    event.preventDefault();

    if (!userId.trim() || Number(userId) <= 0) {
      setRecommendationError("Please enter a valid User ID.");
      return;
    }

    try {
      setRecommendationLoading(true);
      setRecommendationError("");
      setRecommendations([]);

      const response = await fetch(
        `${API_URL}/recommendations/${encodeURIComponent(userId)}?limit=10`
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Unable to load recommendations.");
      }

      setRecommendations(data.recommendations);
    } catch (err) {
      setRecommendationError(err.message);
    } finally {
      setRecommendationLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="navbar">
        <h2>🎬 MovieMate</h2>
        <span>Movie Recommendation System</span>
      </header>

      <section className="hero">
        <p className="subtitle">DISCOVER YOUR NEXT FAVORITE</p>
        <h1>Find Movies You'll Love</h1>
        <p>Explore movies and discover something amazing to watch.</p>

        <form className="search-form" onSubmit={searchMovies}>
          <input
            type="text"
            placeholder="Search for a movie..."
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
          <button type="submit">Search</button>
          <button
            type="button"
            className="reset-button"
            onClick={() => {
              setSearch("");
              fetchMovies();
            }}
          >
            Reset
          </button>
        </form>
      </section>

      <main>
        <section className="movie-section">
          <div className="section-heading">
            <h2>Personalized Recommendations</h2>
            <span>Powered by Machine Learning</span>
          </div>

          <form
            className="search-form"
            onSubmit={fetchRecommendations}
          >
            <input
              type="number"
              min="1"
              placeholder="Enter User ID (e.g. 1)"
              value={userId}
              onChange={(event) => setUserId(event.target.value)}
            />
            <button type="submit" disabled={recommendationLoading}>
              {recommendationLoading ? "Loading..." : "Get Recommendations"}
            </button>
          </form>

          {recommendationLoading && (
            <p className="message">Finding movies for you...</p>
          )}

          {recommendationError && (
            <p className="error">{recommendationError}</p>
          )}

          {!recommendationLoading && recommendations.length > 0 && (
            <>
              <p className="message">
                Top {recommendations.length} recommendations for User {userId}
              </p>

              <div className="movie-grid">
                {recommendations.map((movie) => (
                  <div className="movie-card" key={movie.movie_id}>
                    <div className="movie-icon">⭐</div>
                    <div className="movie-info">
                      <h3>{movie.title}</h3>
                      <p>{movie.genres.replaceAll("|", " • ")}</p>
                      <span>Movie ID: {movie.movie_id}</span>
                      <p>Recommendation Score: {movie.score}</p>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </section>

        <section className="movie-section">
          <div className="section-heading">
            <h2>Explore Movies</h2>
            <span>{movies.length} movies displayed</span>
          </div>

          {loading && <p className="message">Loading movies...</p>}

          {error && <p className="error">{error}</p>}

          {!loading && !error && movies.length === 0 && (
            <p className="message">No movies found.</p>
          )}

          <div className="movie-grid">
            {movies.map((movie) => (
              <div className="movie-card" key={movie.movieId}>
                <div className="movie-icon">🎞️</div>
                <div className="movie-info">
                  <h3>{movie.title}</h3>
                  <p>{movie.genres.replaceAll("|", " • ")}</p>
                  <span>Movie ID: {movie.movieId}</span>
                </div>
              </div>
            ))}
          </div>
        </section>
      </main>

      <footer>
        <p>MovieMate | Built with React, FastAPI & PostgreSQL</p>
      </footer>
    </div>
  );
}

export default App;
