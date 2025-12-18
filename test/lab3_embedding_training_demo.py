import sys
import os
import multiprocessing
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Setup đường dẫn import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.logger import get_logger


class UDCorpus:
    """
    Class này thay thế cho LineSentence.
    Nó giúp đọc file UD (Universal Dependencies) một cách thông minh hơn.
    """

    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        """
        Hàm này sẽ được Word2Vec gọi liên tục để lấy dữ liệu.
        Nó đọc file từng dòng, làm sạch và trả về danh sách các từ.
        """
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 1. Nếu là dòng chứa text gốc (thường bắt đầu bằng '# text = ')
                if line.startswith("# text ="):
                    text = line.replace("# text =", "").strip()
                    # simple_preprocess giúp bỏ dấu câu, lower case, tokenization
                    yield simple_preprocess(text)

                # 2. Nếu file này là raw text (không có metadata)
                # nhưng không phải là dòng trống hoặc comment
                elif line and not line.startswith("#"):
                    # Kiểm tra xem có phải định dạng bảng (CoNLL) không
                    # Nếu dòng có ký tự Tab '\t', khả năng cao là bảng -> Bỏ qua để tránh nhiễu
                    if '\t' in line:
                        continue
                    yield simple_preprocess(line)


def main():
    # 1. Setup Logger
    logger = get_logger("lab3_training.log", output_dir="../output/lab3/")
    logger.info("--- LAB 3 BONUS: TRAIN WORD2VEC (CLEAN DATA) ---")

    # 2. CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU
    data_file = r"..\data\UD_English-EWT\UD_English-EWT\en_ewt-ud-train.txt"

    if not os.path.exists(data_file):
        logger.error(f"Error: Không tìm thấy file tại: {data_file}")
        return

    try:
        # 3. Stream Data & Config
        logger.info(f"\n1. Reading and Cleaning data from: {data_file}")

        sentences = UDCorpus(data_file)

        # Cấu hình Params
        # Giảm vector_size xuống 50 vì tập dữ liệu này thực tế khá nhỏ (~200k từ)
        # Tăng epochs (iter) lên để pretrained_models học kỹ hơn
        params = {
            'vector_size': 50,
            'window': 5,
            'min_count': 2,
            'workers': multiprocessing.cpu_count(),
            'epochs': 10  # Train lặp lại 10 lần
        }
        logger.info(f"   Params: {params}")

        # 4. Train Model
        logger.info("\n2. Training pretrained_models...")
        model = Word2Vec(sentences=sentences, **params)
        logger.info("-> Training completed.")

        # 5. Save Model
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        model_save_path = os.path.join(output_dir, "word2vec_ewt.pretrained_models")
        model.save(model_save_path)
        logger.info(f"-> Model saved to: {model_save_path}")

        # 6. Demo Usage
        logger.info("\n3. Testing the new pretrained_models:")

        test_words = ['government', 'people', 'good', 'time']

        for word in test_words:
            if word in model.wv:
                logger.info(f"\n   Similar to '{word}':")
                # Lấy top 10
                for w, s in model.wv.most_similar(word, topn=10):
                    logger.info(f"     - {w:<15} : {s:.4f}")
            else:
                logger.info(f"\n   Word '{word}' was removed during preprocessing (or not found).")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())

    logger.info("\n--- END ---")


if __name__ == "__main__":
    main()