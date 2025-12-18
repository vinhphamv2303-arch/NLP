import sys
import os

# 1. SETUP MÔI TRƯỜNG
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.logger import get_logger

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col
    from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml import Pipeline
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
except ImportError:
    print("CRITICAL: Chưa cài đặt pyspark. Hãy chạy 'pip install pyspark'")
    sys.exit(1)


def main():
    # 2. CẤU HÌNH LOGGER
    logger = get_logger(filename="lab4_spark_sentiment.log", output_dir="../output/lab4")
    logger.info("=== LAB 4 ADVANCED: SPARK SENTIMENT ANALYSIS ===")

    # 3. KHỞI TẠO SPARK SESSION
    logger.info("Step 1: Initialize Spark Session...")
    spark = SparkSession.builder \
        .appName("Lab4_SparkSentiment") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    # 4. LOAD DATA
    # Đường dẫn tương đối từ thư mục gốc project
    data_path = "../data/sentiments.csv"

    if not os.path.exists(data_path):
        logger.error(f"Lỗi: Không tìm thấy file dữ liệu tại '{data_path}'")
        logger.error("Vui lòng kiểm tra lại đường dẫn file.")
        spark.stop()
        return

    logger.info(f"Step 2: Load Data from {data_path}")
    df = spark.read.csv(data_path, header=True, inferSchema=True)

    # --- XỬ LÝ NHÃN THEO ĐỀ BÀI ---
    # Convert -1/1 labels to 0/1: (sentiment + 1) / 2
    logger.info("Normalizing sentiment labels (-1/1 to 0/1)...")
    df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)

    # Loại bỏ dòng null (nếu có)
    df = df.dropna(subset=["sentiment", "text"])

    # Chia Train/Test (80/20)
    trainingData, testData = df.randomSplit([0.8, 0.2], seed=42)
    logger.info(f"Training Count: {trainingData.count()} | Test Count: {testData.count()}")

    # 5. BUILD PREPROCESSING PIPELINE
    logger.info("Step 3: Build Preprocessing Pipeline...")

    # a. Tokenizer
    tokenizer = Tokenizer(inputCol="text", outputCol="words")

    # b. StopWordsRemover
    stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

    # c. HashingTF (Yêu cầu đề bài: numFeatures=10000)
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)

    # d. IDF
    idf = IDF(inputCol="raw_features", outputCol="features")

    # 6. TRAIN THE MODEL
    logger.info("Step 4: Train the Model (LogisticRegression)...")

    # LogisticRegression (Yêu cầu đề bài: regParam=0.001, maxIter=10)
    lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")

    # Gộp vào Pipeline
    pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, lr])

    # Training
    logger.info("Training started...")
    try:
        model = pipeline.fit(trainingData)
        logger.info("Model training completed.")
    except Exception as e:
        logger.error(f"Training Failed: {e}")
        spark.stop()
        return

    # 7. EVALUATE
    logger.info("Step 5: Evaluate on Test Data...")
    predictions = model.transform(testData)

    # Log một vài kết quả mẫu để kiểm tra
    logger.info("--- PREDICTION SAMPLES ---")
    rows = predictions.select("text", "label", "prediction").limit(10).collect()
    for row in rows:
        check = "CORRECT" if row.label == row.prediction else "WRONG"
        logger.info(f"Text: '{row.text}' | Label: {row.label} | Pred: {row.prediction} -> {check}")

    # Tính Accuracy
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    logger.info("-" * 30)
    logger.info(f"ACCURACY: {accuracy:.4f}")
    logger.info("-" * 30)

    # Đóng Spark
    spark.stop()
    logger.info("=== END LAB 4 ADVANCED ===")


if __name__ == "__main__":
    main()