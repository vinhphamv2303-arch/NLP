from src.core.interfaces import Vectorizer, Tokenizer


class CountVectorizer(Vectorizer):
    """
    A simple CountVectorizer that converts a list of documents into count vectors.
    """

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary_ = {}

    def fit(self, corpus: list):
        self.vocabulary_ = {}
        for doc in corpus:
            tokens = self.tokenizer.tokenize(doc)
            for token in tokens:
                if token not in self.vocabulary_:
                    self.vocabulary_[token] = len(self.vocabulary_)
        return self

    def transform(self, documents: list) -> list:
        vectors = []
        for doc in documents:
            tokens = self.tokenizer.tokenize(doc)
            vector = [0] * len(self.vocabulary_)
            for token in tokens:
                if token in self.vocabulary_:
                    index = self.vocabulary_[token]
                    vector[index] += 1
            vectors.append(vector)
        return vectors