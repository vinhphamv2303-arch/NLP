import sys
import os
import numpy as np

# Setup đường dẫn import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.representations.word_embedder import WordEmbedder
from src.utils.logger import get_logger


def main():
    # Gọi hàm log
    logger = get_logger("lab3_results.log", output_dir="../output/lab3/")

    logger.info("--- LAB 3 TEST RESULTS ---")

    try:
        # 1. Load Model
        logger.info("\n1. Loading Model...")
        embedder = WordEmbedder(model_name='glove-wiki-gigaword-50')
        logger.info("-> Done.")

        # 2. Vector
        word = 'king'
        vec = embedder.get_vector(word)
        logger.info(f"\n2. Vector for '{word}':")
        logger.info(f"   Shape: {vec.shape}")
        logger.info(f"   Values: {vec}")

        # 3. Similarity
        logger.info("\n3. Similarity:")
        logger.info(f"   King - Queen: {embedder.get_similarity('king', 'queen'):.4f}")
        logger.info(f"   King - Man:   {embedder.get_similarity('king', 'man'):.4f}")

        # 4. Most Similar
        logger.info("\n4. Top 5 similar to 'computer':")
        for w, s in embedder.get_most_similar('computer', top_n=10):
            logger.info(f"   {w:<15} : {s:.4f}")

        # 5. Document Embedding
        sent = "The queen rules the country."
        logger.info(f"\n5. Document Vector for: \"{sent}\"")
        doc_vec = embedder.embed_document(sent)

        # In toàn bộ vector
        full_vec = np.array2string(doc_vec, separator=', ', threshold=np.inf)
        logger.info("   Full Vector:")
        logger.info(full_vec)

    except Exception as e:
        logger.error(f"Error: {e}")

    logger.info("\n--- END ---")


if __name__ == "__main__":
    main()