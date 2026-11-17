# 🎬 Movie Recommendation System

A Movie Recommendation System built using Python, Machine Learning, FastAPI, PostgreSQL, and React.

This project recommends movies to users based on their rating history using collaborative filtering techniques.

## 📌 Project Overview

The Movie Recommendation System analyzes movie ratings and provides personalized movie recommendations.

The project includes data exploration, recommendation model development, backend API integration, database management, and a React-based frontend.

## ✨ Features

- Movie dataset exploration and processing
- User-Based Collaborative Filtering
- Item-Based Collaborative Filtering
- Personalized movie recommendations
- Cold-start recommendation handling
- Model evaluation and comparison
- Movie search functionality
- User registration and movie rating APIs
- PostgreSQL database integration
- FastAPI backend
- React frontend

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Collaborative Filtering |
| Data Processing | Pandas, NumPy, SciPy |
| Backend | FastAPI |
| Database | PostgreSQL |
| ORM | SQLAlchemy |
| Migrations | Alembic |
| Frontend | React, JavaScript, CSS |
| Development Tools | Git, GitHub, VS Code |

## 📂 Project Structure

```text
Movie-Recommendation-System/
│
├── backend/
│   ├── app/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── services/
│   │   ├── database.py
│   │   └── main.py
│   ├── migrations/
│   └── seed_movies.py
│
├── frontend/
│   └── src/
│
├── dataset/
│   └── ml-latest-small/
│
├── explore_data.py
├── create_matrix.py
├── recommend_movies.py
├── personalized_recommendations.py
├── user_based_recommendations.py
├── cold_start_recommendations.py
├── evaluate_model.py
├── compare_models.py
├── inspect_movies.py
└── .gitignore
```

## 🤖 Machine Learning Approach

The project explores two collaborative filtering approaches:

**1. Item-Based Collaborative Filtering**

Recommends movies based on similarities between movies rated by users.

**2. User-Based Collaborative Filtering**

Finds users with similar rating patterns and recommends movies they may like.

**3. Cold-Start Handling**

Provides fallback recommendations when sufficient user rating history is not available.

## 📊 Dataset

This project uses the MovieLens Latest Small dataset.

- 9,742 movies
- 100,836 ratings
- 610 users

Dataset folder: `dataset/ml-latest-small/`

## 🚀 Running the Project

### Backend

Navigate to the backend folder:

```bash
cd backend
```

Activate the project's Python virtual environment:

```bash
..\.venv\Scripts\activate
```

Configure the PostgreSQL database connection using your local `.env` file.

Start the FastAPI server:

```bash
python -m uvicorn app.main:app --reload --port 8000
```

Open the API documentation:

http://127.0.0.1:8000/docs

### Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL displayed by Vite in your browser.

## 🔗 Repository

[Movie Recommendation System – GitHub](https://github.com/deekshitha2206-hue/Movie-Recommendation-System)

## 👩‍💻 Author

**Deekshitha Bikkumalla**

GitHub: [@deekshitha2206-hue](https://github.com/deekshitha2206-hue)
