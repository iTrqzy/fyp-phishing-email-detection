from sklearn.feature_extraction.text import TfidfVectorizer


def fit_vectorizer(text, max_features=5000):
    # learns the language from text training
    vectorizer = TfidfVectorizer(max_features=max_features,
                                  ngram_range=(1, 2),
                                  stop_words='english')

    x_train = vectorizer.fit_transform(text)
    return vectorizer, x_train

def transform_vectorizer(vectorizer, text):
    return vectorizer.transform(text)