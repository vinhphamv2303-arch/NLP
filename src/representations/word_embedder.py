import gensim.downloader as api
import numpy as np
from typing import List


class WordEmbedder:
    def __init__(self, model_name: str = 'glove-wiki-gigaword-50'):
        """
        Khởi tạo WordEmbedder và tải mô hình pre-trained.
        """
        print(f"Dang tai mo hinh {model_name}...")
        try:
            self.model = api.load(model_name)
            print("Tai mo hinh thanh cong!")
        except Exception as e:
            print(f"Loi khi tai mo hinh: {e}")
            raise e

    def get_vector(self, word: str) -> np.ndarray:
        """
        Trả về vector embedding cho một từ.
        Nếu từ không có trong từ điển (OOV), trả về vector 0.
        """
        if word in self.model:
            return self.model[word]
        else:
            # Trả về vector 0 nếu từ không tồn tại (Out-of-Vocabulary)
            return np.zeros(self.model.vector_size)

    def get_similarity(self, word1: str, word2: str) -> float:
        """
        Trả về độ tương đồng cosine giữa hai từ.
        """
        if word1 in self.model and word2 in self.model:
            return self.model.similarity(word1, word2)
        else:
            return 0.0

    def get_most_similar(self, word: str, top_n: int = 10):
        """
        Tìm top N từ tương đồng nhất.
        """
        if word in self.model:
            return self.model.most_similar(word, topn=top_n)
        else:
            return []

    def embed_document(self, document: str) -> np.ndarray:
        """
        Tạo vector cho cả văn bản bằng cách lấy trung bình cộng vector các từ.
        """
        # Giả sử dùng tokenizer đơn giản (tách theo khoảng trắng và bỏ ký tự đặc biệt)
        # Nếu bạn có class Tokenizer từ Lab 1, hãy import và dùng ở đây.
        tokens = document.lower().split()
        # Cần xử lý kỹ hơn (bỏ dấu câu) nếu muốn chính xác hơn.

        valid_vectors = []
        for token in tokens:
            # Chỉ lấy vector nếu từ có trong từ điển
            if token in self.model:
                valid_vectors.append(self.model[token])

        if not valid_vectors:
            # Nếu không có từ nào hợp lệ, trả về vector 0
            return np.zeros(self.model.vector_size)

        # Tính trung bình cộng (mean) các vector theo trục dọc
        doc_vector = np.mean(valid_vectors, axis=0)
        return doc_vector