import os
from rag_engine import RAGEngine

def load_and_chunk_markdown(file_path: str) -> list[str]:
    """
    Reads a Markdown file and splits it into logical paragraph chunks
    based on double-newline separations.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Target document not found at: {file_path}")
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split by double newline to separate headers/paragraphs cleanly
    raw_chunks = content.split("\n\n")
    
    # Clean up empty spaces or stray formatting lines
    cleaned_chunks = [chunk.strip() for chunk in raw_chunks if chunk.strip()]
    return cleaned_chunks

def main():
    print("🚀 Starting Ingestion Pipeline for Module 2...")
    
    # 1. Initialize our RAG Engine
    engine = RAGEngine()
    
    # 2. Extract chunks from our newly created knowledge base file
    target_file = "data/knowledge_base/syllabus_mod2.md"
    try:
        chunks = load_and_chunk_markdown(target_file)
        print(f"📄 Found {len(chunks)} structural chunks inside {target_file}")
        
        # 3. Iterate and commit to the persistent database
        for index, chunk in enumerate(chunks):
            doc_id = f"mod2_syllabus_chunk_{index}"
            metadata = {
                "source": target_file,
                "chunk_index": index
            }
            
            print(f"⏳ Processing chunk {index}...")
            engine.add_document(doc_id=doc_id, text=chunk, metadata=metadata)
            
        print("✅ Ingestion complete! All chunks are successfully stored in ChromaDB.")
        
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")

if __name__ == "__main__":
    main()
