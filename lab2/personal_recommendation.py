import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import SVD, Dataset, Reader

def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['Book-Rating'] >= 0]
    valid_books = df['ISBN'].value_counts()[lambda x: x > 1].index
    df = df[df['ISBN'].isin(valid_books)]
    valid_users = df['User-ID'].value_counts()[lambda x: x > 1].index
    df = df[df['User-ID'].isin(valid_users)]
    return df

def books_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    corrections = [
        (209538, "Michael Teitelbaum", 2000, "DK Readers: Creating the X-Men, How It All Began"),
        (220731, "Jean-Marie Gustave", 2003, "Peuple du ciel, suivi de Les Bergers"),
        (221678, "James Buckley", 2000, "DK Readers: Creating the X-Men, How Comic Books Come to Life"),
    ]

    for idx, author, year, title in corrections:
        df.loc[idx, ["Book-Author", "Year-Of-Publication", "Book-Title"]] = [author, year, title]

    df['Year-Of-Publication'] = pd.to_numeric(df['Year-Of-Publication'], errors='coerce')
    df.loc[df['Year-Of-Publication'] > 2024, 'Year-Of-Publication'] = 2024
    df['Year-Of-Publication'] = df['Year-Of-Publication'].astype(int).astype(str)
    df = df.drop([118033, 128890, 129037, 187689])
    df = df.drop(columns=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'])
    return df

def recommend_books(ratings: pd.DataFrame, svd_model: SVD, linreg_model, books: pd.DataFrame) -> pd.DataFrame:
    user_id = ratings[ratings['Book-Rating'] == 0]['User-ID'].value_counts().idxmax()
    zero_ratings_books = ratings[(ratings['User-ID'] == user_id) & (ratings['Book-Rating'] == 0)]

    recommendations = [
        {'ISBN': row['ISBN'], 'svd_rating': svd_model.predict(user_id, row['ISBN']).est}
        for _, row in zero_ratings_books.iterrows()
        if svd_model.predict(user_id, row['ISBN']).est >= 8
    ]

    if not recommendations:
        return pd.DataFrame([])

    recommendations_df = pd.DataFrame(recommendations)
    merged = pd.merge(recommendations_df, books, on='ISBN', how='left')
    merged = merged[merged['Book-Title'].notna() & (merged['Book-Title'] != '')]

    tfidf = TfidfVectorizer(max_features=500)
    title_vectors = tfidf.fit_transform(merged['Book-Title']).toarray()

    merged['Book-Author'] = merged['Book-Author'].astype('category').cat.codes
    merged['Publisher'] = merged['Publisher'].astype('category').cat.codes
    merged['Year-Of-Publication'] = pd.to_numeric(merged['Year-Of-Publication'], errors='coerce')

    features = pd.concat([
        pd.DataFrame(title_vectors, index=merged.index),
        merged[['Book-Author', 'Publisher', 'Year-Of-Publication']]
    ], axis=1)

    features.columns = features.columns.astype(str)
    merged['linreg_rating'] = linreg_model.predict(features)

    return merged[['Book-Title', 'svd_rating', 'linreg_rating']].sort_values(by='linreg_rating', ascending=False)

if __name__ == "__main__":
    ratings = pd.read_csv("Ratings.csv")
    books = pd.read_csv("Books.csv", low_memory=False)
    ratings = ratings_preprocessing(ratings)
    books = books_preprocessing(books)

    with open("svd.pkl", "rb") as svd_file:
        svd_model = pickle.load(svd_file)
    with open("linreg.pkl", "rb") as linreg_file:
        linreg_model = pickle.load(linreg_file)

    recommendations = recommend_books(ratings, svd_model, linreg_model, books)
    print(recommendations)