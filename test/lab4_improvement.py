import sys
import os

os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.logger import get_logger

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col
    from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
    from pyspark.ml.classification import LogisticRegression, NaiveBayes
    from pyspark.ml import Pipeline
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
except ImportError:
    sys.exit(1)


def main():
    logger = get_logger(filename="lab4_improvement.log", output_dir="../output/lab4")
    logger.info("=== TASK 4: FINAL TUNING (LR vs NAIVE BAYES) ===")

    spark = SparkSession.builder.appName("Lab4_Final").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # 1. LOAD DATA
    data_path = "../data/sentiments.csv"
    if not os.path.exists(data_path):
        logger.error("Không tìm thấy data/sentiments.csv")
        return

    df = spark.read.csv(data_path, header=True, inferSchema=True)

    # Xử lý nhãn & Clean nhẹ
    df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)
    df = df.dropna(subset=["sentiment", "text"])

    # Chia Train/Test
    train, test = df.randomSplit([0.8, 0.2], seed=42)

    # 2. PREPROCESSING (Dùng CountVectorizer thay vì HashingTF để NaiveBayes chạy tốt hơn)
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    stopwords = StopWordsRemover(inputCol="words", outputCol="filtered")

    # vocabSize: Giới hạn số từ vựng quan trọng nhất (Feature Selection)
    cv = CountVectorizer(inputCol="filtered", outputCol="raw_features", vocabSize=5000, minDF=2.0)
    idf = IDF(inputCol="raw_features", outputCol="features")

    # --- MODEL 1: LOGISTIC REGRESSION (TUNED) ---
    # Tăng regParam lên 0.02 để giảm overfitting
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20, regParam=0.02)

    # --- MODEL 2: NAIVE BAYES ---
    # Model "huyền thoại" cho text classification
    nb = NaiveBayes(featuresCol="features", labelCol="label", smoothing=1.0, modelType="multinomial")

    # 3. TRAINING & EVALUATE
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

    # --- CHẠY LOGISTIC REGRESSION ---
    logger.info(">>> Running Logistic Regression (Tuned)...")
    pipeline_lr = Pipeline(stages=[tokenizer, stopwords, cv, idf, lr])
    model_lr = pipeline_lr.fit(train)
    pred_lr = model_lr.transform(test)
    acc_lr = evaluator.evaluate(pred_lr)
    logger.info(f"Logistic Regression Accuracy: {acc_lr:.4f}")

    # --- CHẠY NAIVE BAYES ---
    logger.info(">>> Running Naive Bayes...")
    pipeline_nb = Pipeline(stages=[tokenizer, stopwords, cv, idf, nb])
    model_nb = pipeline_nb.fit(train)
    pred_nb = model_nb.transform(test)
    acc_nb = evaluator.evaluate(pred_nb)
    logger.info(f"Naive Bayes Accuracy:        {acc_nb:.4f}")

    # So sánh kết quả
    logger.info("-" * 40)
    logger.info(f"LR Base (Previous): ~0.7295")
    logger.info(f"LR Tuned:           {acc_lr:.4f}")
    logger.info(f"Naive Bayes:        {acc_nb:.4f}")

    winner = "Naive Bayes" if acc_nb > acc_lr else "Logistic Regression"
    logger.info(f"WINNER: {winner}")
    logger.info("-" * 40)

    spark.stop()


if __name__ == "__main__":
    main()