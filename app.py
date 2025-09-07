import streamlit as st
import pickle
import pandas as pd
import requests
import os

def fetch_poster(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=02b34be131731e82c195668ab848e177&language=en-US')
    data = response.json()
    print(data)  # Debugging: See the entire response

    if 'poster_path' in data:
        return "https://image.tmdb.org/t/p/w500" + data['poster_path']
    else:
        return "https://via.placeholder.com/500x750?text=No+Poster+Available"


def recommend(movie):
    movies_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movies_index]
    movies_list_sorted = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movie_posters = []
    recommended_movies = []

    for i in movies_list_sorted:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movie_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_movie_posters


# Download similarity.pkl from Google Drive if not present
SIMILARITY_FILE_ID = "1kRkzQzEoXkeHQh4rC5O9xb35B-XR1Ejw"
if not os.path.exists("similarity.pkl"):
    url = f"https://drive.google.com/uc?export=download&id={SIMILARITY_FILE_ID}"
    r = requests.get(url)
    with open("similarity.pkl", "wb") as f:
        f.write(r.content)

# Load similarity.pkl
with open("similarity.pkl", "rb") as f:
    similarity = pickle.load(f)


# Load movies.pkl
with open("movies.pkl", "rb") as f:
    movies_list = pickle.load(f)
movies = pd.DataFrame(movies_list)

movies_titles = movies_list['title'].values
st.title('Movie Recommender System')

option = st.selectbox(
    "Select a movie:",
    movies_titles,
)

st.write("You selected:", option)

st.button("Reset", type="primary")
if st.button("Recommend"):
    names, posters = recommend(option)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])
