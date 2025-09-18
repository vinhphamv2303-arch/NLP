import os

def load_raw_text_data(file_path: str) -> str:
    """
    Load raw text data from a file.

    Args:
        file_path (str): Path to the text file.

    Returns:
        str: The raw text content of the file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            text = f.read()
        return text
    except Exception as e:
        raise RuntimeError(f"Error reading file {file_path}: {e}")