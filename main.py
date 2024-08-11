from modules import DataPreprocessor, EmbeddingProcessor
from dotenv import dotenv_values
import logging

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
config = dotenv_values(".env")


folder_path = "data"  # Data Folder Path
data_processor = DataPreprocessor(folder_path)
embedding_processor = EmbeddingProcessor()


def add_to_db():
    processed_data = data_processor.load_and_preprocess_data()
    embedding_processor.process_and_store(processed_data)


def search_similar_questions(top_k=5):
    while True:
        similar_questions, similar_answers = (
            embedding_processor.search_similar_questions(
                input("Ask Your Question"), top_k
            )
        )
        for i, question in enumerate(similar_questions):
            print(f"{i+1}. {question}")
        return similar_questions, similar_answers

if __name__ == "__main__":
    add_to_db()
    # search_similar_questions(2)
