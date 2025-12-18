import sys
import os
import shutil

# --- Setup đường dẫn import ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.logger import get_logger

# --- Import PySpark ---
try:
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import Word2Vec
    from pyspark.sql.functions import col, lower, regexp_replace, split, size
except ImportError:
    print("Lỗi: Chưa cài pyspark. Vui lòng chạy 'pip install pyspark'")
    sys.exit(1)


def main():
    # 1. Setup Logger
    logger = get_logger("lab3_spark.log", output_dir="../output/lab3/")
    logger.info("--- LAB 3 ADVANCED: SPARK WORD2VEC (REAL DATA) ---")

    # 2. Khởi tạo Spark Session
    # local[*] dùng tối đa sức mạnh CPU hiện có
    logger.info("1. Initializing Spark Session...")
    try:
        spark = SparkSession.builder \
            .appName("Lab3_Word2Vec_C4") \
            .master("local[*]") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()  # Tăng RAM lên 4g

        spark.sparkContext.setLogLevel("ERROR")
        logger.info("-> Spark Session created successfully.")
    except Exception as e:
        logger.error(f"Failed to start Spark. Error: {e}")
        return

    try:
        # 3. ĐỌC DỮ LIỆU TỪ FILE THẬT
        data_path = r"..\data\c4-train.00000-of-01024-30K.json.gz"

        if not os.path.exists(data_path):
            logger.error(f"Error: Không tìm thấy file tại {data_path}")
            return

        logger.info(f"\n2. Loading data from: {data_path}")

        # Đọc file JSON (Spark xử lý nén tự động)
        df = spark.read.json(data_path)

        # In ra số lượng dòng dữ liệu tìm thấy
        row_count = df.count()
        logger.info(f"-> Loaded {row_count} documents.")

        # 4. Preprocessing
        logger.info("\n3. Preprocessing data...")

        # Pipeline xử lý:
        # 1. col("text"): Lấy cột text
        # 2. lower(): Chữ thường
        # 3. regexp_replace(): Bỏ hết ký tự đặc biệt, chỉ giữ chữ cái và khoảng trắng
        # 4. split(): Tách thành mảng các từ
        processed_df = df.select(
            split(
                regexp_replace(
                    lower(col("text")),
                    "[^a-z\\s]", ""
                ),
                "\\s+"
            ).alias("words")
        )

        # Lọc bỏ các dòng rỗng (size > 1 để tránh mảng chỉ chứa 1 khoảng trắng)
        processed_df = processed_df.filter(size(col("words")) > 1)

        logger.info("-> Preprocessing done.")

        # 5. Train Word2Vec
        logger.info("\n4. Training Word2Vec pretrained_models with Spark...")

        # Cấu hình Word2Vec
        # vectorSize: 100 chiều (chuẩn cho data lớn)
        # minCount: 5 (Bỏ qua từ xuất hiện < 5 lần để giảm nhiễu)
        word2Vec = Word2Vec(
            vectorSize=100,
            minCount=5,
            inputCol="words",
            outputCol="result",
            stepSize=0.025,
            maxIter=3  # Chạy 3 vòng lặp (epochs)
        )

        model = word2Vec.fit(processed_df)
        logger.info("-> Training completed.")

        # 6. Demo Usage
        logger.info("\n5. Testing the pretrained_models:")

        # Thử tìm các từ phổ biến
        test_words = ["computer", "government", "time", "people"]

        for target_word in test_words:
            logger.info(f"\n   Finding synonyms for '{target_word}':")
            try:
                # Tìm 5 từ đồng nghĩa
                synonyms = model.findSynonyms(target_word, 5)

                # Collect kết quả về để in ra log
                rows = synonyms.collect()
                for row in rows:
                    logger.info(f"     - {row['word']:<15} : {row['similarity']:.4f}")
            except Exception:
                logger.info(f"     Word '{target_word}' not found in vocabulary.")

    except Exception as e:
        logger.error(f"Error during Spark Job: {e}")
        import traceback
        logger.error(traceback.format_exc())

    finally:
        # 7. Dọn dẹp
        spark.stop()
        logger.info("\n-> Spark Session stopped.")
        logger.info("--- END ---")


if __name__ == "__main__":
    main()