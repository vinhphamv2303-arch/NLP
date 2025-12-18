import os
from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.core.dataset_loaders import load_raw_text_data


def main():
    # 1. Setup Output Directory & File
    output_dir = "./output/lab1"
    os.makedirs(output_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
    log_file_path = os.path.join(output_dir, "tokenization_log.txt")

    # Mở file log để ghi
    with open(log_file_path, "w", encoding="utf-8") as f:

        # Hàm helper: vừa in ra màn hình, vừa ghi vào file
        def log(message=""):
            print(message)
            f.write(message + "\n")

        log(f"--- LOG START: Output saved to {log_file_path} ---\n")

        # 2. Instantiate Tokenizers
        simple_tokenizer = SimpleTokenizer()
        regex_tokenizer = RegexTokenizer()

        # ==========================================
        # TEST CASE 1: Custom Sentences
        # ==========================================
        log("=== TEST CASE 1: Custom Sentences ===")
        test_sentences = [
            "Hello, world! This is a test.",
            "NLP is fascinating... isn't it?",
            "Let's see how it handles 123 numbers and punctuation!"
        ]

        for sentence in test_sentences:
            log(f"Original: {sentence}")

            # Simple Tokenizer
            s_tokens = simple_tokenizer.tokenize(sentence)
            log(f"  [Simple]: {s_tokens}")

            # Regex Tokenizer
            r_tokens = regex_tokenizer.tokenize(sentence)
            log(f"  [Regex ]: {r_tokens}")
            log("-" * 30)

        # ==========================================
        # TEST CASE 2: UD_English-EWT Dataset
        # ==========================================
        log("\n=== TEST CASE 2: UD_English-EWT Dataset ===")

        # Đường dẫn tuyệt đối
        dataset_path = r"D:\ADMIN\Documents\Classwork\NLP_Lab\data\UD_English-EWT\UD_English-EWT\en_ewt-ud-train.txt"

        try:
            raw_text = load_raw_text_data(dataset_path)

            # Lấy mẫu 500 ký tự đầu
            sample_text = raw_text[:500]

            log("--- Tokenizing Sample Text ---")
            # In mẫu văn bản gốc (cắt ngắn 100 char để đỡ dài dòng log)
            log(f"Original Sample (first 100 chars): {sample_text[:100]}...\n")

            # SimpleTokenizer output
            simple_tokens = simple_tokenizer.tokenize(sample_text)
            log(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}\n")

            # RegexTokenizer output
            regex_tokens = regex_tokenizer.tokenize(sample_text)
            log(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}\n")

            log(f"Total tokens found (Simple): {len(simple_tokens)}")
            log(f"Total tokens found (Regex) : {len(regex_tokens)}")

        except Exception as e:
            log(f"ERROR: Could not load dataset at {dataset_path}")
            log(f"Details: {e}")


if __name__ == "__main__":
    main()