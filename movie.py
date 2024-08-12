import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
movies = pd.read_csv('movies.csv')
print(movies.columns)
ratings = pd.read_csv('ratings.csv')
print(ratings.columns)
# Merge the movies and ratings dataset on movieId
movie_data = pd.merge(ratings, movies, on='movieId')

# Create a pivot table with users as rows and movies as columns
user_movie_matrix = movie_data.pivot_table(index='userId', columns='title', values='rating')

# Fill NaN values with 0 for cosine similarity
user_movie_matrix.fillna(0, inplace=True)
# Compute cosine similarity between movies
movie_similarity = cosine_similarity(user_movie_matrix.T)

# Create a DataFrame of the similarity matrix
movie_similarity_df = pd.DataFrame(movie_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)
def recommend_movies(movie_name, n_recommendations=5):
    # Check if the movie exists in the dataset
    if movie_name not in movie_similarity_df.index:
        return "Movie not found in the dataset"
    
    # Get the similarity scores for the given movie
    similarity_scores = movie_similarity_df[movie_name]
    
    # Sort the movies based on similarity scores and return top n recommendations
    recommended_movies = similarity_scores.sort_values(ascending=False)[1:n_recommendations+1]
    
    return recommended_movies
recommended_movies = recommend_movies('Toy Story (1995)', 5,)
print(recommended_movies)
