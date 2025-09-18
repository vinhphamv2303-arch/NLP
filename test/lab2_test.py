from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer

def main():
    tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer)

    # Sample corpus
    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]

    # Fit and transform
    dt_matrix = vectorizer.fit_transform(corpus)

    # Results
    print("Learned Vocabulary (word -> index):")
    print(vectorizer.vocabulary_)
    print("\nDocument-Term Matrix:")
    for i, vec in enumerate(dt_matrix):
        print(f"Document {i}: {vec}")

if __name__ == "__main__":
    main()