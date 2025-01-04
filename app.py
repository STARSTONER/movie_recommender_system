import streamlit as st
import pickle
import pandas as pd
import requests

def fetch_poster(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=02b34be131731e82c195668ab848e177&language=en-US')
    data = response.json()
    print(data)  # Debugging: See the entire response

    # Check if the 'poster_path' key is in the response
    if 'poster_path' in data:
        return "https://image.tmdb.org/t/p/w500" + data['poster_path']
    else:
        return "https://via.placeholder.com/500x750?text=No+Poster+Available"  # Placeholder image if no poster


def recommend(movie):
    movies_index=movies[movies['title']==movie].index[0]
    distances=similarity[movies_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    recommended_movie_posters=[]

    recommended_movies=[]
    for i in movies_list:
        movie_id=movies.iloc[i[0]].movie_id
        #fetch poster from tmdb using api
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movie_posters.append(fetch_poster(movie_id))
    return recommended_movies,recommended_movie_posters



similarity=pickle.load(open('similarity.pkl','rb'))



movies_list=pickle.load(open('movies.pkl','rb'))
movies=pd.DataFrame(movies_list)

movies_list=movies_list['title'].values
st.title('Movie Recommender System')

option = st.selectbox(
    "How would you like to be contacted?",
    (movies_list),
)

st.write("You selected:", option)

st.button("Reset", type="primary")
if st.button("Recommend"):
    names,posters=recommend(option)

    col1,col2,col3,col4,col5 =st.columns(5)

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

