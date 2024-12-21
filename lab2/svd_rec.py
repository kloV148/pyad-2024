import pandas as pd
import pickle
from surprise import SVD
from surprise import Dataset, Reader
from surprise import accuracy


def preprocess_ratings(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['Book-Rating'] > 0]

    valid_books = df['ISBN'].value_counts()[lambda x: x > 1].index
    df = df[df['ISBN'].isin(valid_books)]

    valid_users = df['User-ID'].value_counts()[lambda x: x > 1].index
    df = df[df['User-ID'].isin(valid_users)]

    df = df.drop_duplicates() 
    return df


def train_and_evaluate_model(ratings: pd.DataFrame) -> None:
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)

    test_data = pd.read_csv("svd_test.csv")
    testset = list(zip(test_data['User-ID'], test_data['ISBN'], test_data['Book-Rating']))

    trainset = data.build_full_trainset()

    svd = SVD(n_factors=50, lr_all=0.005, reg_all=0.02, n_epochs=30, random_state=42)
    svd.fit(trainset)

    predictions = svd.test(testset)
    mae = accuracy.mae(predictions)

    if mae < 1.5:
        with open("svd.pkl", "wb") as file:
            pickle.dump(svd, file)
        print(f"Модель успешно сохранена с MAE: {mae}")
    else:
        print(f"Модель не сохранена. MAE слишком высок: {mae}")


def main():
    ratings = pd.read_csv("Ratings.csv")
    print(ratings.head())
    processed_ratings = preprocess_ratings(ratings)
    train_and_evaluate_model(processed_ratings)


if __name__ == "__main__":
    main()