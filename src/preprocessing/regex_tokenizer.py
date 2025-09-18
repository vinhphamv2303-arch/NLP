'''
class RegexTokenizer that inherits from the Tokenizer interface.
Implement the tokenize method using a single regular expression to extract tokens.
'''
import re

from src.core.interfaces import Tokenizer


class RegexTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        pattern = r'\.{2,}|[!?]{2,}|--+|[+-]?\d+[.,\d]*%?|\w+(?:[-\']\w+)*|[.!?:;,]'

        tokens = re.findall(pattern, text)

        return tokens


if __name__ == "__main__":
    text = "Let's see how it handles 123.0 numbers, and punctuation!"
    tokenizer = RegexTokenizer()
    print(tokenizer.tokenize(text))