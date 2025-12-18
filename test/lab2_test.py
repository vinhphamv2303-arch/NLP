import os
import sys

# Đảm bảo python tìm thấy thư mục src nếu chạy từ thư mục gốc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer


def main():
    # 1. Setup Output Directory & File
    output_dir = "../output/lab2"
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, "vectorization_log.txt")

    # Mở file log để ghi
    with open(log_file_path, "w", encoding="utf-8") as f:

        # Hàm helper: vừa in ra màn hình, vừa ghi vào file
        def log(message=""):
            print(message)
            f.write(message + "\n")

        log(f"--- LOG START: Lab 2 Vectorization Output ---\n")

        # 2. Instantiate Components
        tokenizer = RegexTokenizer()
        vectorizer = CountVectorizer(tokenizer)

        # 3. Define Corpus
        corpus = [
            "I love NLP.",
            "I love programming.",
            "NLP is a subfield of AI."
        ]

        log("--- Input Corpus ---")
        for i, doc in enumerate(corpus):
            log(f"Doc {i}: {doc}")
        log("")

        # 4. Fit and Transform
        dt_matrix = vectorizer.fit_transform(corpus)

        # 5. Log Results
        log("--- Learned Vocabulary (Word -> Index) ---")
        # Sắp xếp vocabulary theo index tăng dần để dễ kiểm tra
        sorted_vocab = sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])
        for word, index in sorted_vocab:
            log(f"  '{word}': {index}")

        log("\n--- Document-Term Matrix ---")
        for i, vec in enumerate(dt_matrix):
            log(f"Document {i}: {vec}")


if __name__ == "__main__":
    main()