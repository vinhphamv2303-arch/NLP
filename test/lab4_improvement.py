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


def evaluate_and_log(predictions, model_name, logger):
    """
    Hàm phụ trợ để tính và log 4 chỉ số quan trọng
    """
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    acc = evaluator.setMetricName("accuracy").evaluate(predictions)
    prec = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    rec = evaluator.setMetricName("weightedRecall").evaluate(predictions)
    f1 = evaluator.setMetricName("f1").evaluate(predictions)

    logger.info(f"--- {model_name} Metrics ---")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")

    return acc, f1


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

    # 2. PREPROCESSING (CountVectorizer + minDF=2.0)
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    stopwords = StopWordsRemover(inputCol="words", outputCol="filtered")

    # vocabSize=5000, minDF=2.0 (Lọc bỏ từ xuất hiện < 2 lần)
    cv = CountVectorizer(inputCol="filtered", outputCol="raw_features", vocabSize=5000, minDF=2.0)
    idf = IDF(inputCol="raw_features", outputCol="features")

    # --- MODEL 1: LOGISTIC REGRESSION (TUNED) ---
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20, regParam=0.02)

    # --- MODEL 2: NAIVE BAYES ---
    nb = NaiveBayes(featuresCol="features", labelCol="label", smoothing=1.0, modelType="multinomial")

    # 3. TRAINING & EVALUATE

    # --- CHẠY LOGISTIC REGRESSION ---
    logger.info(">>> Running Logistic Regression (Tuned)...")
    pipeline_lr = Pipeline(stages=[tokenizer, stopwords, cv, idf, lr])
    model_lr = pipeline_lr.fit(train)
    pred_lr = model_lr.transform(test)

    # Gọi hàm evaluate cho LR
    acc_lr, f1_lr = evaluate_and_log(pred_lr, "Logistic Regression", logger)

    # --- CHẠY NAIVE BAYES ---
    logger.info("\n>>> Running Naive Bayes...")
    pipeline_nb = Pipeline(stages=[tokenizer, stopwords, cv, idf, nb])
    model_nb = pipeline_nb.fit(train)
    pred_nb = model_nb.transform(test)

    # Gọi hàm evaluate cho NB
    acc_nb, f1_nb = evaluate_and_log(pred_nb, "Naive Bayes", logger)

    # --- TỔNG KẾT ---
    logger.info("-" * 40)
    logger.info("COMPARISON SUMMARY:")
    logger.info(f"LR Base (Previous): ~0.7295")
    logger.info(f"LR Tuned Accuracy:  {acc_lr:.4f} (F1: {f1_lr:.4f})")
    logger.info(f"Naive Bayes Accuracy: {acc_nb:.4f} (F1: {f1_nb:.4f})")

    # Quyết định model chiến thắng dựa trên Accuracy (hoặc F1)
    if acc_lr > acc_nb:
        winner = "Logistic Regression"
        diff = acc_lr - acc_nb
    else:
        winner = "Naive Bayes"
        diff = acc_nb - acc_lr

    logger.info(f"WINNER: {winner} (Lead by {diff:.4f})")
    logger.info("-" * 40)

    spark.stop()


if __name__ == "__main__":
    main()