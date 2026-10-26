
# Movie Recommendation System

## Project Overview
This project implements a content-based movie recommendation system using the MovieLens dataset. It recommends movies with similar genres based on the selected movie.

## Dataset
- Dataset: MovieLens
- File: `movies.csv`
- Total movies: 9,742
- Features: Movie ID, title, and genres

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity

## Project Workflow
1. Load the MovieLens dataset.
2. Explore the dataset and check missing and duplicate values.
3. Clean movie titles and genre information.
4. Visualize the most common movie genres.
5. Convert genre information into numerical features using TF-IDF.
6. Calculate cosine similarity between movies.
7. Recommend the top 10 movies with similar genres.

## Results
The dataset contains 9,742 movies and 24 TF-IDF features were generated for the genre-based recommendation model.

The system was tested with:
- Toy Story (1995)
- Jumanji (1995)
- Titanic (1997)
- Heat (1995)

The system successfully generated 10 recommendations for each tested movie.

## How to Run

### 1. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Run the project
```bash
python movie_recommendation.py
```

Make sure `movies.csv` is in the project folder.

## Limitations
Recommendations are based on movie genres only. Movies with identical genres can receive the same similarity score. The system does not currently use movie plots, actors, directors, or user ratings.
