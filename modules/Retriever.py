from sentence_transformers import SentenceTransformer
import chromadb
import logging


class EmbeddingProcessor:

    def __init__(self, model_name="BAAI/bge-m3", db_path="./Vector_DB"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = SentenceTransformer(model_name)
        self.logger.info(f"Loaded model {model_name}.")
        # chromadb setup
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        # collection setup
        self.collection = self.chroma_client.get_or_create_collection(
            name="questions_embeddings"
        )
        self.logger.info(f"Connected to Chroma DB at {db_path}.")

    def calculate_embeddings(self, df):
        self.logger.info(f"Calculating embeddings for {len(df)} questions.")
        questions = df["question"].tolist()
        embeddings = self.model.encode(questions)
        return embeddings

    def store_embeddings_in_chroma(self, df, embeddings):
        self.logger.info(f"Storing embeddings in Chroma DB.")

        for i, (question, embedding) in enumerate(zip(df["question"], embeddings)):
            self.collection.add(
                documents=[question],
                embeddings=[embedding.tolist()],
                metadatas=[{"answer": df["answer"][i]}],
                ids=[str(self.collection.count() + 1)],
            )
        self.logger.info(
            f"Total {self.collection.count()} embeddings stored in Chroma DB."
        )

    def process_and_store(self, df):
        self.logger.info("Processing data and storing embeddings.")
        embeddings = self.calculate_embeddings(df)
        self.store_embeddings_in_chroma(df, embeddings)

    def search_similar_questions(self, question, top_k=5):
        self.logger.info(f"Searching for similar questions to: {question}")
        embedding = self.model.encode([question])
        results = self.collection.query(
            query_embeddings=embedding.tolist(), n_results=top_k
        )

        # بازگرداندن نتایج به صورت لیستی از سوالات و جواب‌های مشابه
        similar_questions = results["documents"][0]
        similar_answers = [metadata["answer"] for metadata in results["metadatas"][0]]
        return similar_questions, similar_answers
