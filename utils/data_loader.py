import pandas as pd
import numpy as np
import streamlit as st
import os
import pickle
import time

# Auto-extraction logic at import time
def initialize_data_files():
    """Ensure data/movies.csv and data/ratings.csv exist by extracting or mocking."""
    os.makedirs("data", exist_ok=True)
    
    # 1. Extract movies.csv from movies.pkl if missing
    if not os.path.exists("data/movies.csv"):
        if os.path.exists("movies.pkl"):
            try:
                with open("movies.pkl", "rb") as f:
                    df_pkl = pickle.load(f)
                
                # Rename columns to match what's expected by load_movies
                df_csv = df_pkl.rename(columns={
                    "movie_id": "movieId",
                    "avg_user_rating": "rating_avg",
                    "num_user_ratings": "rating_count"
                })
                
                # If year is float in movies.pkl, ensure it's loaded as float
                if "year" in df_csv.columns:
                    df_csv["year"] = df_csv["year"].astype(float)
                
                # Save the complete metadata df so that we can load full descriptions
                df_csv.to_csv("data/movies.csv", index=False)
            except Exception as e:
                # If loading pickle fails, we let fallback handle it
                pass
        
    # 2. Generate ratings.csv linking to loaded movieIds if missing
    if not os.path.exists("data/ratings.csv"):
        try:
            if os.path.exists("data/movies.csv"):
                movies_df = pd.read_csv("data/movies.csv")
                movie_ids = movies_df["movieId"].tolist()
            else:
                movie_ids = list(range(1, 201))
            
            np.random.seed(42)
            n_users = 150
            ratings_data = []
            for user_id in range(1, n_users + 1):
                # Pick a random subset of movies for each user to rate
                num_ratings = np.random.randint(5, 30)
                selected_movies = np.random.choice(movie_ids, size=min(num_ratings, len(movie_ids)), replace=False)
                for movie_id in selected_movies:
                    ratings_data.append({
                        "userId": int(user_id),
                        "movieId": int(movie_id),
                        "rating": float(np.round(np.random.uniform(2.0, 5.0) * 2) / 2),
                        "timestamp": int(time.time() - np.random.randint(0, 365*24*3600))
                    })
            ratings_df = pd.DataFrame(ratings_data)
            ratings_df.to_csv("data/ratings.csv", index=False)
        except Exception as e:
            pass

# Initialize files
initialize_data_files()

@st.cache_data
def load_movies():
    """Load movies dataset. Returns DataFrame with columns:
    movieId, title, genres, year, rating_avg, rating_count, popularity, etc."""
    try:
        df = pd.read_csv("data/movies.csv")
        # Parse year from title if needed: "Toy Story (1995)" → 1995
        if "year" not in df.columns or df["year"].isnull().all():
            df["year"] = df["title"].str.extract(r'\((\d{4})\)').astype(float)
        
        # Ensure title_clean exists
        if "title_clean" not in df.columns:
            df["title_clean"] = df["title"].str.replace(r'\s*\(\d{4}\)', '', regex=True)
            
        # Ensure other columns match the target schemas
        if "rating_avg" not in df.columns and "avg_user_rating" in df.columns:
            df["rating_avg"] = df["avg_user_rating"]
        if "rating_count" not in df.columns and "num_user_ratings" in df.columns:
            df["rating_count"] = df["num_user_ratings"]
            
        return df
    except FileNotFoundError:
        return _mock_movies()

@st.cache_data
def load_ratings():
    try:
        return pd.read_csv("data/ratings.csv")
    except FileNotFoundError:
        return _mock_ratings()

def _mock_movies():
    """Generate 200 mock movies for fallback mode."""
    genres_list = ["Drama","Action","Comedy","Thriller","Sci-Fi","Romance","Horror","Animation","Documentary","Crime"]
    np.random.seed(42)
    n = 200
    years = np.random.randint(1990, 2024, n)
    return pd.DataFrame({
        "movieId": range(1, n+1),
        "title_clean": [f"Movie {i}" for i in range(1, n+1)],
        "title": [f"Movie {i} ({int(y)})" for i, y in enumerate(years)],
        "year": years.astype(float),
        "genres": np.random.choice(genres_list, n),
        "rating_avg": np.round(np.random.uniform(5.5, 9.5, n), 1),
        "rating_count": np.random.randint(100, 50000, n),
        "popularity": np.random.uniform(0, 100, n),
        "overview": [f"This is the description for movie {i}." for i in range(1, n+1)]
    })

def _mock_ratings():
    """Generate 1000 mock ratings for fallback mode."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "userId": np.random.randint(1, 101, n),
        "movieId": np.random.randint(1, 201, n),
        "rating": np.round(np.random.uniform(1.0, 5.0) * 2) / 2,
        "timestamp": np.random.randint(1609459200, 1704067200, n)
    })
