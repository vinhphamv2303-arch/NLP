"""
class SimpleTokenizer that inherits from the Tokenizer interface.
Implement the tokenize method:
– Convert the text to lowercase.
– Split the text into tokens based on whitespace.
– Handle basic punctuation (e.g. ',', '.', '?', '!') by splitting them from words. 
For example, "Hello, world!" should become ["hello", ",", "world", "!"].
"""
from src.core.interfaces import Tokenizer


class SimpleTokenizer(Tokenizer):
    def __init__(self):
        self.word_punctuations = ['.', ',', '?', '!', ':', ';', '(', ')', '"', "'"]

    def tokenize(self, text: str) -> list[str]:
        # Convert to lowercase
        text = text.lower()
        tokens = []

        for word in text.split():
            # If the mark appears at the end of a word or sentence, it’s punctuation.
            # Else it's a numeric symbol.
            for punct in self.word_punctuations:
                if word[-1] == punct:
                    parts = word.split(punct)
                    tokens.append(parts[0])
                    tokens.append(punct)
                    break
            else:
                tokens.append(word)

        return tokens


if __name__ == "__main__":
    tokenizer = SimpleTokenizer()
    text = "Hello, world! This is a test."
    print(tokenizer.tokenize(text))