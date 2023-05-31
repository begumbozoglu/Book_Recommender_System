
#### APP.PY FOUR YOUTUBE ###
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from PIL import Image
from youtubesearchpython import VideosSearch
import base64

# Veriyi ve model bileşenlerini yükleme
df = pd.read_pickle("preprocessed_books.pkl")
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open("indices.pkl", "rb") as f:
    indices = pickle.load(f)
cosine_sim = np.load('cosine_sim.npy')


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpeg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
        unsafe_allow_html=True
    )


add_bg_from_local('background_image.jpeg')


def search_youtube_videos(query):
    videos_search = VideosSearch(query, limit=1)
    results = videos_search.result()["result"]

    if results:
        video_url = f"https://www.youtube.com/watch?v={results[0]['id']}"
        return {"title": results[0]['title'], "url": video_url}
    else:
        return None


def recommend_books_t(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:2]
    book_indices = [i[0] for i in sim_scores]
    return df.iloc[book_indices]


# Streamlit uygulamasını başlatma
st.title("📚MIUULIB📚")
st.write(
    "MIUULIB is a content-based book recommendation system. If you enjoyed the content of the book you're reading and"
    " would like to explore related books, simply choose the last book you read from the list below. Additionally, "
    "we will provide you with a link to access the book and share video content from YouTube users discussing the book."
    "Keep on reading!!!")

# Önceden belirlenmiş kitap listesi
book_list = sorted(df['TITLE'].tolist())

# Kullanıcıdan kitap seçimi için bir selectbox oluştur
book_title = st.selectbox("Select the Last Book You Read:", book_list)

# Giriş varsa, kitap önerilerini ve ilgili YouTube videosunu gösterme
if st.button('Recommend'):
    if book_title:
        recommended_books = recommend_books_t(book_title)
    for i in range(len(recommended_books)):
        st.write("Book:")
        image = Image.open(requests.get(recommended_books.iloc[i]['BOOK_IMAGE'], stream=True).raw)
        st.image(image, width=150)
        st.write("Title:", recommended_books.iloc[i]['TITLE'])
        st.write("Author:", recommended_books.iloc[i]['AUTHOR'])
        st.write("---------------------------------------")
        st.write("Description:", recommended_books.iloc[i]['DESCRIPTION'])
        st.write("Genre:", ', '.join(recommended_books.iloc[i]['GENRE']))

        st.markdown(f'See on GoodReads: [Show Me The Book🔥]({recommended_books.iloc[i]["BOOK_LINK"]})')

        st.write("---------------------------------------")

        # YouTube'da ilgili video araması yapma
        search_term = recommended_books.iloc[i]['TITLE'] + " " + recommended_books.iloc[i]['AUTHOR'] + " Book Analysis"
        video = search_youtube_videos(search_term)
        if video:
            try:
                st.video(video['url'])
            except Exception as e:
                st.write(
                    f"Error: {e}. Can't play the video here, but you can watch it on YouTube [here]({video['url']}).")
        else:
            st.write("No related YouTube video found.")

