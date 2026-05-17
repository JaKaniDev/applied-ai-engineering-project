import os
import chromadb
from google import genai
from google.genai import types


class RAGEngine:
    def __init__(self, db_path="db/chroma_local", collection_name="knowledge_library"):
        """
        Initializes the persistent ChromaDB client and sets up the Gemini GenAI client.
        """
        # 1. Initialize Persistent Chroma Client
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

        # 2. Initialize Gemini GenAI Client (Pulls GEMINI_API_KEY from env)
        self.ai_client = genai.Client()
        # Note: We use the modern 'text-embedding-004' model for high-fidelity vector mapping
        self.embedding_model = "gemini-embedding-2"

    def get_embedding(self, text: str) -> list[float]:
        """
        Generates a vector embedding for a given string using the Gemini API.
        """
        response = self.ai_client.models.embed_content(
            model=self.embedding_model,
            contents=text
        )
        # Extract the list of floats representing the vector coordinates
        return response.embeddings[0].values

    def add_document(self, doc_id: str, text: str, metadata: dict = None):
        """
        Generates an embedding for a document chunk and stores it in ChromaDB.
        """
        vector = self.get_embedding(text)
        self.collection.add(
            ids=[doc_id],
            embeddings=[vector],
            documents=[text],
            metadatas=[metadata or {}]
        )
        print(f"Successfully indexed document chunk: {doc_id}")

    def query_knowledge(self, query_text: str, n_results: int = 2) -> list[str]:
        """
        Converts a user query into a vector, searches ChromaDB, and returns the top matching text chunks.
        """
        query_vector = self.get_embedding(query_text)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results
        )
        # Extract and return the underlying raw document texts
        return results['documents'][0] if results['documents'] else []


if __name__ == "__main__":
    # Smoke test to verify everything initializes perfectly
    print("Initializing RAG Engine architectural components...")
    engine = RAGEngine()
    print("Initialization complete. Core engine structure is verified.")