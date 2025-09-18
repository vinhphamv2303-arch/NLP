from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.core.dataset_loaders import load_raw_text_data

# def main():
#     simple_tokenizer = SimpleTokenizer()
#     regex_tokenizer = RegexTokenizer()
#
#     test_cases = [
#         "Hello, world! This is a test.",
#         "NLP is fascinating... isn't it?",
#         "Let's see how it handles 123 numbers and punctuation!"
#     ]
#
#     print("=== Testing SimpleTokenizer ===")
#     for sentence in test_cases:
#         tokens = simple_tokenizer.tokenize(sentence)
#         print(f"Sentence: {sentence}")
#         print(f"Tokens: {tokens}")
#         print()
#
#     print("=== Testing RegexTokenizer ===")
#     for sentence in test_cases:
#         tokens = regex_tokenizer.tokenize(sentence)
#         print(f"Sentence: {sentence}")
#         print(f"Tokens: {tokens}")
#         print()

def main():
    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()

    # UD_English-EWT dataset
    dataset_path = r"D:\ADMIN\Documents\Classwork\NLP_Lab\data\UD_English-EWT\UD_English-EWT\en_ewt-ud-train.txt"
    raw_text = load_raw_text_data(dataset_path)

    sample_text = raw_text[:500] # First 500 characters
    print("\n--- Tokenizing Sample Text from UD_English-EWT ---")
    print(f"Original Sample (first 100 chars): {sample_text[:100]}...\n")

    # SimpleTokenizer
    simple_tokens = simple_tokenizer.tokenize(sample_text)
    print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}\n")

    # RegexTokenizer
    regex_tokens = regex_tokenizer.tokenize(sample_text)
    print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}\n")

if __name__ == "__main__":
    main()