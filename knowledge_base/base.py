import sqlite3
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import time
import os

class RAGKnowledgeBase:
    """
    A RAG (Retrieval-Augmented Generation) knowledge base supporting
    both sparse (BM25) and dense (tensor) retrieval, designed for
    handling a large number of chunks and using a database and vector index.
    """

    def __init__(self, db_path="rag_kb.db", embedding_model="all-mpnet-base-v2", faiss_index_path="faiss.index", reset_db=False):
        """
        Initializes the RAGKnowledgeBase.

        Args:
            db_path (str, optional): Path to the SQLite database file. Defaults to "rag_kb.db".
            embedding_model (str, optional): The name of the SentenceTransformer model. Defaults to "all-mpnet-base-v2".
            faiss_index_path (str, optional): Path to save/load the FAISS index. Defaults to "faiss.index".
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        if reset_db:
            # delete db and index
            os.remove(db_path)
            os.remove(faiss_index_path)

        # Create the table if it doesn't exist
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                embedding BLOB  -- Store embedding as BLOB
            )
        """)
        self.conn.commit()

        # Sparse Retrieval (BM25)
        self.bm25 = None
        self._build_bm25_index()

        # Dense Retrieval (Tensor)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.faiss_index_path = faiss_index_path
        self.faiss_index = self._load_faiss_index()

    def _build_bm25_index(self):
        """Builds or updates the BM25 index from the database."""
        self.cursor.execute("SELECT content FROM chunks")
        documents = [row[0] for row in self.cursor.fetchall()]
        if documents:
            tokenized_docs = [doc.lower().split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized_docs)
        else:
            self.bm25 = None

    def _load_faiss_index(self):
        """Loads the FAISS index from disk or creates a new one."""
        try:
            index = faiss.read_index(self.faiss_index_path)
            return index
        except RuntimeError:  # File not found or corrupted
            # Create a new index if it doesn't exist or is corrupted
            embedding_size = self.embedding_model.get_sentence_embedding_dimension()
            index = faiss.IndexFlatL2(embedding_size)  
            self._update_faiss_index()
            return index

    def _update_faiss_index(self):
        """Updates the FAISS index with embeddings from the database."""
        self.cursor.execute("SELECT embedding FROM chunks")
        embeddings_blob = [row[0] for row in self.cursor.fetchall()]
        if embeddings_blob:
            embeddings = np.array([np.frombuffer(emb, dtype=np.float32) for emb in embeddings_blob])
            self.faiss_index.add(embeddings)
            faiss.write_index(self.faiss_index, self.faiss_index_path)

    def add_documents(self, documents):
        """
        Adds a list of documents to the knowledge base, chunking them
        and storing them in the database with embeddings.

        Args:
            documents (list): A list of document strings.
        """
        for doc in documents:
            for chunk in doc["chunks"]:
                chunk = chunk.strip()
                if chunk:
                    embedding = self.embedding_model.encode(chunk, convert_to_numpy=True)
                    self.cursor.execute("INSERT INTO chunks (content, embedding) VALUES (?, ?)",
                                        (chunk, embedding.tobytes()))
        self.conn.commit()
        self._build_bm25_index()
        self._update_faiss_index()

    def search(self, query, top_k=5, retrieval_method="sparse"):
        """
        Searches the knowledge base for relevant chunks.

        Args:
            query (str): The search query.
            top_k (int, optional): The number of top chunks to retrieve. Defaults to 5.
            retrieval_method (str, optional): The retrieval method to use.
                                             Options: "sparse" (BM25), "dense" (tensor).
                                             Defaults to "sparse".

        Returns:
            list: A list of tuples, where each tuple contains the chunk ID and its score/distance.
                  The list is sorted by score/distance (descending for BM25, ascending for FAISS).
        """
        if retrieval_method == "sparse":
            if not self.bm25:
                return []
            tokenized_query = query.lower().split()
            doc_scores = self.bm25.get_scores(tokenized_query)
            # Fetch chunk IDs based on BM25 scores (order matters)
            indexed_scores = sorted(enumerate(doc_scores), key=lambda x: x[1], reverse=True)[:top_k]
            results = []
            for doc_index, score in indexed_scores:
                self.cursor.execute("SELECT id FROM chunks LIMIT 1 OFFSET ?", (doc_index,))
                row = self.cursor.fetchone()
                if row:
                    results.append((row[0], score))
            return results

        elif retrieval_method == "dense":
            query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            if self.faiss_index.ntotal == 0:
                return []
            distances, indices = self.faiss_index.search(np.array([query_embedding]), top_k)
            results = []
            for i in range(len(indices[0])):
                if indices[0][i] != -1:  # Check if a valid index was found
                    # Need to map the FAISS index back to the database chunk ID
                    # This requires storing the mapping or re-fetching based on embedding
                    # For simplicity, we re-fetch based on the embedding (can be optimized)
                    index_id = int(indices[0][i]) 
                    embedding_to_find = self.faiss_index.reconstruct(index_id).tobytes()
                    self.cursor.execute("SELECT id FROM chunks WHERE embedding = ?", (embedding_to_find,))
                    row = self.cursor.fetchone()
                    if row:
                        results.append((row[0], distances[0][i]))
            return sorted(results, key=lambda x: x[1]) # Sort by distance (lower is better for FAISS)
        else:
            raise ValueError(f"Invalid retrieval method: {retrieval_method}. Choose 'sparse' or 'dense'.")

    def get_chunk_by_id(self, chunk_id):
        """
        Retrieves a chunk by its ID.

        Args:
            chunk_id: The ID of the chunk.

        Returns:
            str or None: The chunk text if found, otherwise None.
        """
        self.cursor.execute("SELECT content FROM chunks WHERE id = ?", (chunk_id,))
        row = self.cursor.fetchone()
        return row[0] if row else None

    def close(self):
        """Closes the database connection."""
        self.conn.close()


if __name__ == "__main__":
    # Initialize the knowledge base
    knowledge_base = RAGKnowledgeBase()

    documents = [
        {
            "id": "doc1",
            "chunks": ["The capital of France is Paris. It's a beautiful city.", "London is the capital of England, and a major global hub."],
            "metadata": {"source": "https://example.com/france", "author": "John Doe"}
        },
        {
            "id": "doc2",
            "chunks": ["Germany is famous for its beer and cars. Berlin is the capital."],
            "metadata": {"source": "https://example.com/germany", "author": "Jane Smith"}
        }
    ]

    knowledge_base.add_documents(documents)

    # Sparse Retrieval (BM25)
    query = "capital cities"
    sparse_results = knowledge_base.search(query, retrieval_method="sparse")
    print("Sparse Retrieval Results (BM25):")
    start_time = time.time()
    for chunk_id, score in sparse_results:
        print(f"ID: {chunk_id}, Score: {score:.4f}, Chunk: {knowledge_base.get_chunk_by_id(chunk_id)}")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.4f} seconds")

    # Dense Retrieval (Tensor)
    dense_results = knowledge_base.search(query, retrieval_method="dense")
    print("\nDense Retrieval Results (Tensor):")
    start_time = time.time()
    for chunk_id, distance in dense_results:
        print(f"ID: {chunk_id}, Distance: {distance:.4f}, Chunk: {knowledge_base.get_chunk_by_id(chunk_id)}")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.4f} seconds")

    # Search with a different query
    query = "programming languages applications"
    dense_results_programming = knowledge_base.search(query, retrieval_method="dense")
    print(f"\nDense Retrieval Results for '{query}':")
    start_time = time.time()
    for chunk_id, distance in dense_results_programming:
        print(f"ID: {chunk_id}, Distance: {distance:.4f}, Chunk: {knowledge_base.get_chunk_by_id(chunk_id)}")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.4f} seconds")

    # Close the database connection
    knowledge_base.close()
    
    
    