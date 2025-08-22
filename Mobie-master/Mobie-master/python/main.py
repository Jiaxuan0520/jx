import streamlit as st
import numpy as np
import pandas as pd
import pickle
import requests
from timeit import default_timer as timer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

movies = pickle.load(open("../pickle/movies.pkl", "rb"))
cosine = pickle.load(open("../pickle/cosine_sim.pkl", "rb"))
knn_similarity = pickle.load(open("../pickle/knn.pkl", "rb"))
lsa = pickle.load(open("../pickle/lsa.pkl", "rb"))

# emoji icon link: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Mobie", page_icon=":movie_camera:", layout= "wide")

st.title("Welcome to MOBIE")

st.header("Movie Recommender System")
selected_alg = st.selectbox("Select algorithm", ("Cosine Similarity", "K-Nearest Neighbours", "Latent Semantic Analysis"))

selected_movies = st.selectbox("Select movies from dropdown", movies)

# Cosine Similarity
def recommendMovieCosine(selected_movies):
    start = timer()

    # Find the index of the movie that matches the title
    index = movies[movies["Series_Title"] == selected_movies].index[0]
    # Sort movies by similarity score (distance) in descending order(get high similarity score)
    distance = sorted(list(enumerate(cosine[index])), reverse=True, key=lambda vector: vector[1])

    end = timer()
    # Get the top 10 most similar movies
    recommended_movies_name = [movies.iloc[i[0]]["Series_Title"] 
    for i in distance[0:10]]

    recommended_movies_img = [movies.iloc[i[0]]["Poster_Link"] 
    for i in distance[0:10]]

    time_executed = (end - start) * 1000
    return recommended_movies_name, recommended_movies_img, time_executed


# KNN
def recommendMovieKNN(selected_movies):

    # Start the timer
    start = timer()

    index = movies[movies["Series_Title"] == selected_movies].index[0]
    
    # Reduce the dimensionality of the input movie
    pca = PCA(n_components=1000)  

    # Apply PCA to reduce dimensionality
    similarity_reduced = pca.fit_transform(knn_similarity)  
    movie_reduced = pca.transform(knn_similarity[index].reshape(1, -1))
    
    # Perform the k-nearest neighbors search
    knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=20)
    knn.fit(similarity_reduced)

    distances, indices = knn.kneighbors(movie_reduced, n_neighbors=10)
    indices = indices.flatten()  # Flatten the indices to use for Pandas indexing

    #Stop the timer
    end = timer()

    #Retrieve the titles and poster links of the top-k recommended movies
    recommended_movies_name = movies.iloc[indices]["Series_Title"].values[:10]
    recommended_movies_img = movies.iloc[indices]["Poster_Link"].values[:10]

    #Calculate the time taken for the execution
    time_executed = (end - start) * 1000

    #Display the movie name, movie poster and time executed
    return recommended_movies_name, recommended_movies_img, time_executed

# LSA
def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

def find_similar_movies(target_vec, lsa):
    distances = []
    for idx, movie_vec in enumerate(lsa):
        diff = euclidean_distance(target_vec, movie_vec)
        distances.append((idx, diff))
    
    # Sort distances in ascending order
    distances.sort(key=lambda x: x[1])
    
    # Return similar movies
    return distances[:10]

def recommendMovieLSA(selected_movies):
    start = timer()

    # Find the index of the movie that matches the title
    index = movies[movies["Series_Title"] == selected_movies].index[0]

    # Perform similarity search on the full dataset
    target_vec = lsa[index]
    similar_movies = find_similar_movies(target_vec, lsa)

    end = timer()

    # Get the top 10 most similar movies
    recommended_movies_name = [movies.iloc[i[0]]["Series_Title"] 
    for i in similar_movies[0:10]]

    recommended_movies_img = [movies.iloc[i[0]]["Poster_Link"] 
    for i in similar_movies[0:10]]

    time_executed = (end - start) * 1000
    return recommended_movies_name, recommended_movies_img, time_executed

#-----------------------------------------------------------------
# Measure performance
# Precision: measures how many of the top recommended items are relevant
def precision_at_k(same_series, y_true, y_pred, k):
    relevant = len(set(y_true) & set(y_pred[:k]))  # Intersection of ground truth and predicted
    if (relevant == same_series.shape[0]) & (same_series.shape[0] != 0):  
        return relevant / same_series.shape[0]
    return relevant / k

# Recall: measures how many relevant items are captured in the top recommendation
def recall_at_k(same_series, y_true, y_pred, k):
    relevant = len(set(y_true) & set(y_pred[:k]))
    if len(y_true) == 0: 
        return 0
    return relevant / len(y_true) 

# harmonic mean of precision and recall
def f1_at_k(prec, rec, y_true, y_pred, k):
    if prec + rec == 0:
        return 0
    return 2 * (prec * rec) / (prec + rec)

def chk_performance(selected_movies, similar_movies):
    same_series = movies[movies["Series_Title"].str.contains(selected_movies, case=False, na=False)]

    # movie must contain
    y_true = same_series["Series_Title"]

    # movies predicted
    y_pred = similar_movies

    k = 10

    precision = precision_at_k(same_series, y_true, y_pred, k)
    recall = recall_at_k(same_series, y_true, y_pred, k)
    f1 = f1_at_k(precision, recall, y_true, y_pred, k)

    return precision, recall, f1

#---------------
# Display result
if st.button("Show Recommendations"):
    if(selected_alg == "Cosine Similarity"):
        movies_name, movies_img, time_executed = recommendMovieCosine(selected_movies)
    elif(selected_alg == "K-Nearest Neighbours"):
        movies_name, movies_img, time_executed = recommendMovieKNN(selected_movies)
    else:
        movies_name, movies_img, time_executed = recommendMovieLSA(selected_movies)
        
    #precision, recall, f1 = chk_performance(selected_movies, movies_name)
        
    st.markdown("#")
    col1, col2, col3, col4, col5 = st.columns(5)
    st.markdown("#")
    col6, col7, col8, col9, col10 = st.columns(5)
    with col1:
        st.image(movies_img[0], use_column_width=True)
        st.text(movies_name[0])
    with col2:
        st.image(movies_img[1], use_column_width=True)
        st.text(movies_name[1])
    with col3:
        st.image(movies_img[2], use_column_width=True)
        st.text(movies_name[2])
    with col4:
        st.image(movies_img[3], use_column_width=True)
        st.text(movies_name[3])
    with col5:
        st.image(movies_img[4], use_column_width=True)
        st.text(movies_name[4])
    with col6:
        st.image(movies_img[5], use_column_width=True)
        st.text(movies_name[5])
    with col7:
        st.image(movies_img[6], use_column_width=True)
        st.text(movies_name[6])
    with col8:
        st.image(movies_img[7], use_column_width=True)
        st.text(movies_name[7])
    with col9:
        st.image(movies_img[8], use_column_width=True)
        st.text(movies_name[8])
    with col10:
        st.image(movies_img[9], use_column_width=True)
        st.text(movies_name[9])
    
    st.markdown("*")
    st.text(f"Time Executed: {time_executed:.2f} ms")
    # st.text(f"Precision: {precision:.2f}")
    # st.text(f"Recall: {recall:.2f}")
    # st.text(f"F1: {f1:.2f}")