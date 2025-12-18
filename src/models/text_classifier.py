import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List, Dict, Any


class TextClassifier:
    def __init__(self, vectorizer: Any):
        """
        Khởi tạo bộ phân loại.
        :param vectorizer: Một instance của Vectorizer (Tfidf hoặc Count)
                           đã được implement ở Lab trước hoặc dùng sklearn.
        """
        self.vectorizer = vectorizer
        # Sử dụng solver='liblinear' như yêu cầu bài lab cho dataset nhỏ
        self._model = LogisticRegression(solver='liblinear')

    def fit(self, texts: List[str], labels: List[int]):
        """
        Huấn luyện mô hình.
        1. Vector hóa văn bản đầu vào.
        2. Train model Logistic Regression trên các vector đó.
        """
        # Biến đổi text thành ma trận đặc trưng (Feature Matrix)
        X = self.vectorizer.fit_transform(texts)

        # Huấn luyện model
        self._model.fit(X, labels)
        print("Model training completed.")

    def predict(self, texts: List[str]) -> List[int]:
        """
        Dự đoán nhãn cho văn bản mới.
        """
        # Chỉ dùng transform (không fit lại) để đảm bảo nhất quán feature
        X = self.vectorizer.transform(texts)

        # Dự đoán
        predictions = self._model.predict(X)
        return predictions.tolist()

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """
        Đánh giá hiệu năng mô hình bằng các chỉ số.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0)
        }
        return metrics