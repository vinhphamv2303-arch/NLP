from abc import ABC, abstractmethod

# Abstract class cho Tokenizer
class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        pass


# Abstract class cho Vectorizer
class Vectorizer(ABC):
    @abstractmethod
    def fit(self, corpus: list):
        pass

    @abstractmethod
    def transform(self, documents: list) -> list:
        pass

    def fit_transform(self, corpus: list) -> list:
        self.fit(corpus)
        return self.transform(corpus)