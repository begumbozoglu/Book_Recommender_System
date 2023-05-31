import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from unidecode import unidecode
import chardet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import pickle

# Setting display options for pandas dataframes
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Loading the dataset
df = pd.read_csv("GoodReads_100k_books.csv", low_memory=False)

# Renaming and reordering the columns
df.columns = map(str.upper, df.columns)
df = df.rename(columns={"DESC": "DESCRIPTION",
                        "IMG": "BOOK_IMAGE",
                        "LINK": "BOOK_LINK",
                        "TOTALRATINGS": "TOTAL_RATINGS",
                        "BOOKFORMAT": "BOOK_FORMAT"})

df = df[['ISBN', 'TITLE', 'AUTHOR', 'DESCRIPTION', 'GENRE', 'PAGES', 'BOOK_FORMAT', 'BOOK_IMAGE', 'BOOK_LINK',
         'ISBN13', 'REVIEWS', 'RATING', 'TOTAL_RATINGS']]
df.head()
df.isnull().sum()
# Handling missing values
df["ISBN"] = df["ISBN"].fillna("unknown")
df["GENRE"] = df["GENRE"].fillna("unknown")
df["BOOK_FORMAT"] = df["BOOK_FORMAT"].fillna("unknown")
df["BOOK_IMAGE"] = df["BOOK_IMAGE"].fillna("unknown")
df.dropna(subset="DESCRIPTION", inplace=True)
df.drop("ISBN13", axis=1, inplace=True)

# Splitting the genre into a list of genres
df['GENRE'] = df['GENRE'].apply(lambda x: x.split(', ') if isinstance(x, str) else ['unknown'])


# Encoding check for titles
def detect_encoding(data: str) -> str:
    result = chardet.detect(data[:50].encode())  # checking the first 50 characters
    return result['encoding']


df["IS_VALID_UTF8"] = df["TITLE"].apply(detect_encoding) == "utf-8"
invalid_utf8_titles = df[df['IS_VALID_UTF8'] == True]['TITLE']

# Dropping rows with invalid utf-8 titles
df = df.drop(invalid_utf8_titles.index).reset_index(drop=True)
df = df.drop(columns=['IS_VALID_UTF8'])

# Encoding check for descriptions
df["IS_VALID_UTF8_CONTROL_desc"] = df["DESCRIPTION"].apply(detect_encoding) == "utf-8"
invalid_utf8_titles_desc = df[df['IS_VALID_UTF8_CONTROL_desc'] == True]['DESCRIPTION']

# Dropping rows with invalid utf-8 descriptions
df = df.drop(invalid_utf8_titles_desc.index).reset_index(drop=True)
df = df.drop(columns=["IS_VALID_UTF8_CONTROL_desc"])

df = df.loc[df["TOTAL_RATINGS%"] >= 700].reset_index()

# Veriyi pickle dosyası olarak kaydetme
df.to_pickle("preprocessed_books.pkl")

df.groupby('AUTHOR')["TOTAL_RATINGS"].sum().sort_values(ascending=False).head(20)
df.groupby(['TITLE', "AUTHOR"])["TOTAL_RATINGS"].sum().sort_values(ascending=False).head(20)

df[df["AUTHOR"].str.contains("Agatha Christie")].sort_values(by="RATING", ascending=False).head(10)

# Merging description and genre into one column
df['DESCRIPTION_AND_GENRE'] = df['DESCRIPTION'] + df['GENRE'].apply(', '.join)

# Content-based filtering
tfidf = TfidfVectorizer(stop_words='english')
df['DESCRIPTION_AND_GENRE'] = df['DESCRIPTION_AND_GENRE'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['DESCRIPTION_AND_GENRE'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df['TITLE']).drop_duplicates()

# Model bileşenlerini pickle dosyalarına kaydetme
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
with open("indices.pkl", "wb") as f:
    pickle.dump(indices, f)
np.save('cosine_sim.npy', cosine_sim)


def recommend_books_t(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]
    book_indices = [i[0] for i in sim_scores]
    return df.iloc[book_indices]


df.head()

df.columns

print("The Kite Runner" in df['TITLE'].values)
print("Khaled Hosseini" in df['AUTHOR'].values)

df[df["AUTHOR"].str.contains("Anthony Burgess")]
df[df["TITLE"].str.contains("A Clash of Kings")]

recommend_books_t("A Clash of Kings")

df["BOOK_IMAGE_URL"][0]
