from sklearn.feature_extraction.text import TfidfVectorizer


def fit_vectorizer(texts, max_features=5000):
    # Learns vocabulary on training text only
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
    )
    x_train = vectorizer.fit_transform(texts)
    return vectorizer, x_train


def transform_vectorizer(vectorizer, texts):
    return vectorizer.transform(texts)
