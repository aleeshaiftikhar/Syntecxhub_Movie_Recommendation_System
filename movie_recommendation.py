import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("movies.csv")

print("Dataset loaded successfully!\n")
print("First five movies:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())
print("\nDataset information:")
df.info()
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nFirst 10 movie genres:")
print(df["genres"].head(10))

print("\n--- Data Cleaning ---")

df = df.drop_duplicates(subset="movieId")

df = df.dropna(subset=["title", "genres"])

df["title"] = df["title"].str.strip()
df["genres"] = df["genres"].str.strip()

df = df[df["title"] != ""]

print("Dataset shape after cleaning:", df.shape)
print("Missing values after cleaning:")
print(df[["title", "genres"]].isnull().sum())
genre_counts = (
    df["genres"]
    .str.split("|")
    .explode()
    .value_counts()
)

genre_counts = genre_counts.drop(
    labels=["(no genres listed)"],
    errors="ignore"
)

print("\nTop 10 genres:")
print(genre_counts.head(10))
plt.figure(figsize=(10, 6))
genre_counts.head(10).sort_values().plot(kind="barh")

plt.title("Top 10 Movie Genres")
plt.xlabel("Number of Movies")
plt.ylabel("Genre")
plt.tight_layout()

plt.savefig("top_genres.png", dpi=300)
plt.show()

print("\nGenre chart saved as top_genres.png")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("\n--- Building Recommendation Model ---")
df["genres_clean"] = df["genres"].str.replace("|", " ", regex=False)
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df["genres_clean"])

print("TF-IDF matrix shape:", tfidf_matrix.shape)
cosine_sim = cosine_similarity(tfidf_matrix, dense_output=False)

print("Cosine similarity matrix shape:", cosine_sim.shape)
print("Recommendation model built successfully!")

df = df.reset_index(drop=True)

def recommend_movies(movie_title, num_recommendations=10):
    # Search for the movie by title
    matches = df[
        df["title"].str.lower().str.contains(
            movie_title.lower(), regex=False
        )
    ]

    if matches.empty:
        print(f"\nMovie '{movie_title}' was not found in the dataset.")
        return

    movie_index = matches.index[0]

    similarity_scores = list(
        enumerate(cosine_sim[movie_index].toarray().flatten())
    )
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = [
        item for item in similarity_scores
        if item[0] != movie_index
    ]

    top_movies = similarity_scores[:num_recommendations]

    print(f"\nMovies similar to '{df.iloc[movie_index]['title']}':\n")

    for rank, (index, score) in enumerate(top_movies, start=1):
        title = df.iloc[index]["title"]
        genres = df.iloc[index]["genres"]

        print(
            f"{rank}. {title} | Genres: {genres} "
            f"| Similarity: {score:.2f}"
        )

    return top_movies

# Test recommendations for different movies
recommend_movies("Toy Story (1995)")
recommend_movies("Jumanji (1995)")
recommend_movies("Titanic (1997)")
recommend_movies("Heat (1995)")


