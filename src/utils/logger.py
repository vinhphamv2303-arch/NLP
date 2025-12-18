import logging
import os
import sys

def get_logger(filename: str, output_dir: str = "./output/lab3/"):
    # 1. Tạo thư mục
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, filename)

    # 2. Khởi tạo logger
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)

    # 3. Reset handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # 4. Định dạng: CHỈ HIỆN NỘI DUNG (%(message)s)
    # Không hiện thời gian hay level
    formatter = logging.Formatter('%(message)s')

    # Handler File
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)

    # Handler Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger