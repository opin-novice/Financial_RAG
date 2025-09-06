#!/usr/bin/env python3
"""
🔍 Quick Chunk Viewer for ChromaDB
==================================
Simple utility to view and search chunks in the ChromaDB database
"""

import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration
CHROMA_DB_PATH = Path("chroma_db")
EMBEDDING_MODEL = "BAAI/bge-m3"

def load_vectorstore():
    """Load ChromaDB vectorstore"""
    if not CHROMA_DB_PATH.exists():
        print(f"❌ ChromaDB database not found at {CHROMA_DB_PATH}")
        return None
    
    print("[INFO] Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu",
            "trust_remote_code": True
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )
    
    return Chroma(
        persist_directory=str(CHROMA_DB_PATH),
        embedding_function=embeddings
    )

def main():
    """Main function."""
    print("🔍 Quick Chunk Viewer")
    print("=" * 40)
    
    # Load vectorstore
    print("[INFO] Loading ChromaDB...")
    vectorstore = load_vectorstore()
    
    if not vectorstore:
        print("❌ Failed to load ChromaDB")
        return
    
    # Get collection info
    collection = vectorstore._collection
    if not collection:
        print("❌ No collection found in ChromaDB")
        return
    
    count = collection.count()
    print(f"[INFO] Found {count} chunks in the database")
    print("=" * 40)
    
    # Get all documents
    results = collection.get()
    if not results or 'documents' not in results:
        print("❌ No documents found in collection")
        return
    
    documents = results['documents']
    metadatas = results.get('metadatas', [])
    
    # Show first few chunks
    for i in range(min(10, len(documents))):  # Show first 10 chunks
        doc = documents[i]
        metadata = metadatas[i] if i < len(metadatas) else {}
        
        print(f"\n📄 Chunk #{i+1}:")
        print(f"   Source: {metadata.get('source', 'unknown')}")
        print(f"   Size: {len(doc)} characters")
        print(f"   Content: {doc[:200]}{'...' if len(doc) > 200 else ''}")
        print(f"   Metadata: {metadata}")
        print("-" * 60)
    
    if len(documents) > 10:
        print(f"\n... and {len(documents) - 10} more chunks")
        print(f"\nTo see all chunks, use: python inspect_chroma_chunks.py --detailed")
    
    # Interactive search
    print(f"\n🔍 Try searching your chunks:")
    while True:
        try:
            query = input("\nEnter search query (or 'quit' to exit): ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if query:
                results = vectorstore.similarity_search(query, k=3)
                print(f"\n📋 Top 3 results for '{query}':")
                for i, doc in enumerate(results, 1):
                    print(f"\n  {i}. Source: {doc.metadata.get('source', 'unknown')}")
                    print(f"     Preview: {doc.page_content[:150]}{'...' if len(doc.page_content) > 150 else ''}")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
