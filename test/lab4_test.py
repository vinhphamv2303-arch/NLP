import sys
import os
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import train_test_split
from src.representations.count_vectorizer import CountVectorizer
from src.models.text_classifier import TextClassifier
from src.utils.logger import get_logger
from src.core.interfaces import Tokenizer


class SimpleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list:
        text = text.lower()
        return re.findall(r'\b\w+\b', text)


def main():
    logger = get_logger(filename="lab4_results.log", output_dir="../output/lab4")
    logger.info("=== BẮT ĐẦU LAB 4: TEXT CLASSIFICATION ===")

    # --- SỬA DATASET: Đảm bảo từ vựng được lặp lại ---
    texts = [
        # POSITIVE (Label 1)
        "This movie is fantastic and I love it!",
        "The acting was superb, a truly great experience.",
        "Highly recommend this, a masterpiece.",
        "I really enjoyed the plot.",
        "Love this film, it is great.",  # Thêm từ 'love', 'great'

        # NEGATIVE (Label 0)
        "I hate this film, it's terrible.",
        "What a waste of time, absolutely boring.",
        "Could not finish watching, so bad.",
        "The script was awful and dull.",
        "I hate the acting, it is bad."  # Thêm câu chứa 'hate' và 'bad'
    ]
    # 5 Pos, 5 Neg
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    # Chia Train/Test (Test size 20% = 2 mẫu)
    # random_state=10: Giúp chia đều hơn cho ví dụ nhỏ này
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=10
    )

    logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # KHỞI TẠO
    try:
        my_tokenizer = SimpleTokenizer()
        vectorizer = CountVectorizer(tokenizer=my_tokenizer)
        classifier = TextClassifier(vectorizer)
    except Exception as e:
        logger.error(f"Lỗi khởi tạo: {e}")
        return

    # HUẤN LUYỆN
    logger.info("Đang training...")
    classifier.fit(X_train, y_train)

    # DỰ ĐOÁN
    y_pred = classifier.predict(X_test)

    logger.info("\n--- KẾT QUẢ ---")
    correct_count = 0
    for text, pred, true in zip(X_test, y_pred, y_test):
        sentiment = "Positive" if pred == 1 else "Negative"
        check = "ĐÚNG" if pred == true else "SAI"
        if pred == true: correct_count += 1
        logger.info(f"Câu: '{text}'")
        logger.info(f"   -> Dự đoán: {sentiment} | Gốc: {true} -> {check}")

    # ĐÁNH GIÁ
    metrics = classifier.evaluate(y_test, y_pred)
    logger.info("\n--- METRICS ---")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")

    logger.info("\n=== HOÀN THÀNH ===")


if __name__ == "__main__":
    main()