#!/usr/bin/env python3
"""
RAG Answer Generator for Evaluation (Using Advanced RAG from main2.py)
Generates answers for questions in CSV file using the advanced RRF fusion RAG system
"""
import os
import csv
import json
import time
import sys
import logging
from typing import Dict, List, Tuple

# Import the advanced RAG system components from main2.py
# We need to import the classes and configuration from main2.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration from main2.py
from main2 import (
    CoreRAGSystem,
    FAISS_INDEX_PATH,
    EMBEDDING_MODEL,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_TEMPERATURE,
    OLLAMA_TOP_P,
    OLLAMA_NUM_PREDICT,
    OLLAMA_REPEAT_PENALTY,
    MAX_DOCS_FOR_RETRIEVAL,
    MAX_DOCS_FOR_CONTEXT
)

# Setup logging to match main2.py
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGAnswerGenerator:
    """Generate answers using Advanced RAG system from main2.py for evaluation purposes"""
    
    def __init__(self):
        """Initialize Advanced RAG components"""
        logger.info("🚀 Initializing Advanced RAG Answer Generator...")
        self._init_rag()
        
    def _init_rag(self):
        """Initialize Advanced RAG system components"""
        try:
            # Initialize the CoreRAGSystem from main2.py
            logger.info("🔧 Initializing CoreRAGSystem from main2.py...")
            self.rag_system = CoreRAGSystem()
            logger.info("✅ Advanced RAG system initialized successfully!")
            
            # Display system info
            try:
                info = self.rag_system.get_system_info()
                logger.info("📊 Advanced RAG System Status:")
                logger.info(f"  - Documents: {info.get('total_vectors', 0):,}")
                logger.info(f"  - FAISS Index: {'✅' if info.get('faiss_index_loaded', False) else '❌'}")
                logger.info(f"  - Cross-Encoder: {'✅' if info.get('cross_encoder_loaded', False) else '❌'}")
                logger.info(f"  - LLM: {'✅' if info.get('llm_initialized', False) else '❌'}")
                logger.info(f"  - Embedding Model: {EMBEDDING_MODEL}")
                logger.info(f"  - LLM Model: {OLLAMA_MODEL}")
                logger.info(f"  - Max Docs Retrieved: {MAX_DOCS_FOR_RETRIEVAL}")
                logger.info(f"  - Max Docs for Context: {MAX_DOCS_FOR_CONTEXT}")
            except Exception as e:
                logger.warning(f"Could not get system info: {e}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Advanced RAG system: {e}")
            raise e
    
    def generate_answer(self, question: str) -> Dict:
        """
        Generate answer for a question using Advanced RAG system
        
        Args:
            question: The question to answer
            
        Returns:
            Dictionary with answer, context, and metadata
        """
        logger.info(f"🔍 Processing question: {question[:100]}...")
        start_time = time.time()
        
        try:
            # Use the advanced RAG system to process the query
            result = self.rag_system.process_query_sync(question)
            
            # Extract information from the advanced RAG result
            answer = result.get("response", "")
            contexts = result.get("contexts", [])
            sources = result.get("sources", [])
            detected_language = result.get("detected_language", "unknown")
            retrieval_method = result.get("retrieval_method", "rrf_fusion")
            documents_found = result.get("documents_found", 0)
            documents_used = result.get("documents_used", 0)
            
            processing_time = time.time() - start_time
            
            # Format sources for compatibility
            source_files = []
            if sources:
                for source in sources:
                    if isinstance(source, dict):
                        source_files.append(source.get("file", "Unknown"))
                    else:
                        source_files.append(str(source))
            
            response = {
                "question": question,
                "predicted_answer": answer.strip() if answer else "",
                "context": contexts,
                "sources": source_files,
                "detected_language": detected_language,
                "retrieval_method": retrieval_method,
                "num_docs_retrieved": documents_found,
                "num_docs_used": documents_used,
                "processing_time": round(processing_time, 2),
                "advanced_rag": True  # Flag to indicate this uses advanced RAG
            }
            
            logger.info(f"✅ Advanced RAG answer generated in {processing_time:.2f}s")
            logger.info(f"📊 Retrieved: {documents_found} docs, Used: {documents_used} docs, Method: {retrieval_method}")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return {
                "question": question,
                "predicted_answer": f"Error generating answer: {str(e)}",
                "context": [],
                "sources": [],
                "detected_language": "unknown",
                "retrieval_method": "error",
                "num_docs_retrieved": 0,
                "num_docs_used": 0,
                "processing_time": 0.0,
                "advanced_rag": True,
                "error": str(e)
            }
    
    def process_csv_file(self, csv_file: str, output_file: str = None) -> List[Dict]:
        """
        Process questions from CSV file and generate answers
        
        Args:
            csv_file: Path to CSV file with questions
            output_file: Optional path to save results as JSON
            
        Returns:
            List of dictionaries with questions, answers, and context
        """
        if not os.path.exists(csv_file):
            print(f"[ERROR] CSV file not found: {csv_file}")
            return []
        
        print(f"[INFO] 📖 Loading questions from: {csv_file}")
        
        results = []
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                questions = list(reader)
            
            logger.info(f"Found {len(questions)} questions to process")
            
            for i, row in enumerate(questions, 1):
                question = row.get('question', '').strip()
                reference_answer = row.get('reference_answer', '').strip()
                document_source = row.get('document_source', '').strip()
                
                if not question:
                    logger.warning(f"Empty question at row {i}, skipping...")
                    continue
                
                logger.info(f"📝 Processing question {i}/{len(questions)}")
                
                # Generate answer using Advanced RAG
                rag_result = self.generate_answer(question)
                
                # Combine with reference data - include all advanced RAG metadata
                evaluation_item = {
                    "question": question,
                    "predicted_answer": rag_result["predicted_answer"],
                    "reference_answer": reference_answer,
                    "context": rag_result["context"],
                    "sources": rag_result["sources"],
                    "document_source": document_source,
                    "detected_language": rag_result.get("detected_language", "unknown"),
                    "retrieval_method": rag_result.get("retrieval_method", "rrf_fusion"),
                    "num_docs_retrieved": rag_result.get("num_docs_retrieved", 0),
                    "num_docs_used": rag_result.get("num_docs_used", 0),
                    "processing_time": rag_result["processing_time"],
                    "advanced_rag": rag_result.get("advanced_rag", True)
                }
                
                # Include error information if present
                if "error" in rag_result:
                    evaluation_item["error"] = rag_result["error"]
                
                results.append(evaluation_item)
                
                # Small delay to avoid overwhelming the system
                time.sleep(0.5)
            
            logger.info(f"✅ Processed {len(results)} questions successfully using Advanced RAG")
            
            # Save results if output file specified
            if output_file:
                self._save_results(results, output_file)
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Failed to process CSV file: {e}")
            return []
    
    def _save_results(self, results: List[Dict], output_file: str):
        """Save results to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

def main():
    """Main function to generate answers for evaluation using Advanced RAG"""
    print("=" * 80)
    print("ADVANCED RAG ANSWER GENERATOR FOR EVALUATION")
    print("=" * 80)
    
    # Initialize generator
    try:
        generator = RAGAnswerGenerator()
    except Exception as e:
        print(f"❌ Failed to initialize Advanced RAG system: {e}")
        print("Please ensure:")
        print("1. FAISS index exists in 'faiss_index' directory")
        print("2. Ollama is running with llama3.2:3b model")
        print("3. All required models are downloaded")
        return
    
    # Input and output files
    csv_file = "car_loan_qa.csv"
    output_file = "car_loan_evaluation_data.json"
    
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        print("Please ensure the CSV file exists with columns: question, reference_answer, document_source")
        return
    
    print(f"📊 Processing questions from: {csv_file}")
    print(f"💾 Will save results to: {output_file}")
    print("🚀 Using Advanced RAG with RRF Fusion (FAISS + BM25 + ColBERT + Dirichlet)")
    print("-" * 80)
    
    # Process all questions
    results = generator.process_csv_file(csv_file, output_file)
    
    if results:
        print("\n" + "=" * 80)
        print("ADVANCED RAG PROCESSING SUMMARY")
        print("=" * 80)
        print(f"📊 Total questions processed: {len(results)}")
        print(f"⏱️  Average processing time: {sum(r['processing_time'] for r in results) / len(results):.2f}s")
        print(f"📚 Average documents retrieved: {sum(r.get('num_docs_retrieved', 0) for r in results) / len(results):.1f}")
        print(f"🎯 Average documents used: {sum(r.get('num_docs_used', 0) for r in results) / len(results):.1f}")
        
        # Show retrieval method statistics
        retrieval_methods = {}
        languages = {}
        for r in results:
            method = r.get('retrieval_method', 'unknown')
            retrieval_methods[method] = retrieval_methods.get(method, 0) + 1
            lang = r.get('detected_language', 'unknown')
            languages[lang] = languages.get(lang, 0) + 1
        
        print(f"🔍 Retrieval methods: {dict(retrieval_methods)}")
        print(f"🌐 Languages detected: {dict(languages)}")
        
        # Show sample result
        if results:
            sample = results[0]
            print(f"\n📝 Sample Advanced RAG Result:")
            print(f"   Question: {sample['question'][:100]}...")
            print(f"   Predicted: {sample['predicted_answer'][:100]}...")
            print(f"   Reference: {sample['reference_answer'][:100]}...")
            print(f"   Context docs: {len(sample.get('context', []))}")
            print(f"   Language: {sample.get('detected_language', 'unknown')}")
            print(f"   Method: {sample.get('retrieval_method', 'unknown')}")
        
        print(f"\n💾 Results saved to: {output_file}")
        print("🎯 Ready for cosine similarity evaluation with Advanced RAG data!")
    else:
        print("❌ No results generated")
    
    print("\n" + "=" * 80)
    print("ADVANCED RAG ANSWER GENERATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
