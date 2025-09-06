#!/usr/bin/env python3
"""
RAG Test Selector - Interactive JSON File Selection for RAG Evaluation
=====================================================================
Allows you to select any JSON file from the dataqa folder to test main2.py
and produces evaluation results similar to cosev2.py and delmax.py
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Import your RAG system
from main2 import CoreRAGSystem

# Configuration
EMBEDDING_MODEL = "BAAI/bge-m3"
DATAQA_FOLDER = "dataqa"
LOGS_FOLDER = "logs"

# Set up logging
os.makedirs(LOGS_FOLDER, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{LOGS_FOLDER}/rag_test_selector.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class EvaluationMetrics:
    """Container for evaluation metrics"""
    def __init__(self, question_id: int, question: str, ground_truth: str):
        self.question_id = question_id
        self.question = question
        self.ground_truth = ground_truth
        self.llm_response = ""
        self.response_similarity = 0.0
        self.metadata = {}
        self.processing_time = 0.0
        self.error = None

class RAGTestSelector:
    """Interactive RAG testing with JSON file selection"""
    
    def __init__(self):
        """Initialize the test selector"""
        logger.info("🚀 Initializing RAG Test Selector...")
        self._init_embedding_model()
        self._init_rag_system()
        logger.info("✅ RAG Test Selector initialized successfully")
    
    def _init_embedding_model(self):
        """Initialize the embedding model"""
        try:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def _init_rag_system(self):
        """Initialize the RAG system"""
        try:
            logger.info("Initializing Core RAG system...")
            self.rag_system = CoreRAGSystem()
            
            # Simple check: process_query must exist
            if not hasattr(self.rag_system, 'process_query') or not callable(self.rag_system.process_query):
                raise Exception("RAG system does not have process_query method")

            logger.info("✅ RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise

    def get_available_json_files(self) -> List[Path]:
        """Get list of available JSON files in dataqa folder"""
        dataqa_path = Path(DATAQA_FOLDER)
        if not dataqa_path.exists():
            logger.error(f"DataQA folder not found: {DATAQA_FOLDER}")
            return []
        
        json_files = list(dataqa_path.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files in {DATAQA_FOLDER}")
        return json_files

    def display_file_selection(self, json_files: List[Path]) -> Path:
        """Display interactive file selection menu"""
        print("\n" + "="*60)
        print("📁 Available JSON Files in dataqa folder:")
        print("="*60)
        
        for i, file_path in enumerate(json_files, 1):
            file_size = file_path.stat().st_size / 1024  # Size in KB
            print(f"{i:2d}. {file_path.name} ({file_size:.1f} KB)")
        
        print("="*60)
        
        while True:
            try:
                choice = input(f"Select a file (1-{len(json_files)}) or 'q' to quit: ").strip()
                
                if choice.lower() in ['q', 'quit', 'exit']:
                    print("👋 Goodbye!")
                    sys.exit(0)
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(json_files):
                    selected_file = json_files[choice_num - 1]
                    print(f"✅ Selected: {selected_file.name}")
                    return selected_file
                else:
                    print(f"❌ Invalid choice. Please enter a number between 1 and {len(json_files)}")
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                sys.exit(0)

    def load_ground_truth(self, file_path: Path) -> List[Dict]:
        """Load ground truth data from JSON file"""
        if not file_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"✅ Loaded {len(data)} questions from {file_path.name}")
        return data

    def get_rag_response(self, question: str) -> Tuple[str, Dict]:
        """Get LLM response for a question using the RAG system"""
        try:
            result = self.rag_system.process_query(question)
            llm_response = result.get('response', '')
            
            metadata = {
                'num_docs': result.get('num_docs', 0),
                'processing_time': result.get('processing_time', 0.0),
                'sources': result.get('sources', []),
                'detected_language': result.get('detected_language', 'unknown'),
                'language_confidence': result.get('language_confidence', 0.0),
                'retrieval_method': result.get('retrieval_method', 'unknown'),
                'documents_found': result.get('documents_found', 0),
                'documents_used': result.get('documents_used', 0),
                'cross_encoder_used': result.get('cross_encoder_used', False)
            }
            return llm_response, metadata
        except Exception as e:
            logger.error(f"Failed to get RAG response: {e}")
            return "", {}

    def compute_embeddings(self, text: str) -> np.ndarray:
        """Compute embeddings for text"""
        if not text or not text.strip():
            return np.array([])
        return self.embedding_model.encode([text.strip()], normalize_embeddings=True, show_progress_bar=False)

    def evaluate_single_question(self, question_data: Dict, question_id: int) -> EvaluationMetrics:
        """Evaluate a single question"""
        question = question_data['question']
        ground_truth_answer = question_data.get('answer', question_data.get('predicted_answer', ''))
        metrics = EvaluationMetrics(question_id, question, ground_truth_answer)
        
        try:
            start_time = datetime.now()
            llm_response, metadata = self.get_rag_response(question)
            metrics.llm_response = llm_response
            metrics.metadata = metadata
            metrics.processing_time = (datetime.now() - start_time).total_seconds()
            
            # Evaluate LLM response against ground truth
            if llm_response and ground_truth_answer:
                llm_embeddings = self.compute_embeddings(llm_response)
                ground_truth_embeddings = self.compute_embeddings(ground_truth_answer)
                if llm_embeddings.size > 0 and ground_truth_embeddings.size > 0:
                    response_similarity = cosine_similarity(llm_embeddings, ground_truth_embeddings)[0][0]
                    metrics.response_similarity = float(response_similarity)
                    logger.info(f"Question {question_id}: LLM Response Similarity = {response_similarity:.4f}")
            else:
                metrics.error = "No LLM response or ground truth available"
            
            return metrics
        except Exception as e:
            metrics.error = str(e)
            return metrics

    def run_evaluation(self, ground_truth_file: Path, max_questions: int = None) -> str:
        """Run evaluation on selected file"""
        print(f"\n🔍 Running evaluation on: {ground_truth_file.name}")
        
        ground_truth_data = self.load_ground_truth(ground_truth_file)
        
        if max_questions:
            ground_truth_data = ground_truth_data[:max_questions]
            print(f"📊 Evaluating {len(ground_truth_data)} questions (limited to {max_questions})")
        else:
            print(f"📊 Evaluating all {len(ground_truth_data)} questions")
        
        results = []
        successful = 0
        failed = 0
        
        for i, qdata in enumerate(ground_truth_data):
            print(f"\n📝 Processing question {i+1}/{len(ground_truth_data)}: {qdata['question'][:50]}...")
            
            metrics = self.evaluate_single_question(qdata, i)
            results.append({
                'question_id': metrics.question_id,
                'question': metrics.question,
                'ground_truth': metrics.ground_truth,
                'llm_response': metrics.llm_response,
                'response_similarity': metrics.response_similarity,
                'metadata': metrics.metadata,
                'processing_time': metrics.processing_time,
                'error': metrics.error
            })
            
            if metrics.error:
                failed += 1
            else:
                successful += 1
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = ground_truth_file.stem
        output_file = f"{LOGS_FOLDER}/rag_test_{filename_base}_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'test_file': str(ground_truth_file),
                'total_questions': len(ground_truth_data),
                'successful': successful,
                'failed': failed,
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Results saved to: {output_file}")
        return output_file

    def calculate_statistics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate statistics for response similarity values"""
        similarities = []
        
        for result in results:
            if 'response_similarity' in result and result['response_similarity'] is not None:
                similarities.append(float(result['response_similarity']))
        
        if not similarities:
            return {}
        
        stats = {
            'mean': np.mean(similarities),
            'median': np.median(similarities),
            'std': np.std(similarities),
            'min': np.min(similarities),
            'max': np.max(similarities),
            'count': len(similarities)
        }
        
        return stats

    def print_results(self, stats: Dict[str, float], file_path: str, total_questions: int, successful: int, failed: int):
        """Print formatted results"""
        print("\n" + "="*60)
        print("📈 RAG EVALUATION RESULTS")
        print(f"📄 Test File: {file_path}")
        print("="*60)
        
        print(f"📊 Total Questions: {total_questions}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print()
        
        if not stats:
            print("❌ No valid similarity data to analyze")
            return
        
        print(f"📈 Mean Similarity:     {stats['mean']:.4f}")
        print(f"📈 Median Similarity:   {stats['median']:.4f}")
        print(f"📈 Std Dev:             {stats['std']:.4f}")
        print(f"📈 Min Similarity:      {stats['min']:.4f}")
        print(f"📈 Max Similarity:      {stats['max']:.4f}")
        print()
        
        # Quality assessment
        mean_score = stats['mean']
        if mean_score >= 0.8:
            quality = "🟢 Excellent"
        elif mean_score >= 0.6:
            quality = "🟡 Good"
        elif mean_score >= 0.4:
            quality = "🟠 Fair"
        elif mean_score >= 0.2:
            quality = "🔴 Poor"
        else:
            quality = "🔴 Very Poor"
        
        print(f"🎯 Overall Quality: {quality} (Mean: {mean_score:.4f})")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Interactive RAG Test Selector")
    parser.add_argument('--file', '-f', help='Specific JSON file to test (optional)')
    parser.add_argument('--max-questions', '-m', type=int, help='Maximum number of questions to test')
    parser.add_argument('--non-interactive', '-n', action='store_true', help='Non-interactive mode')
    
    args = parser.parse_args()
    
    try:
        selector = RAGTestSelector()
        
        if args.file:
            # Use specified file
            file_path = Path(args.file)
            if not file_path.exists():
                print(f"❌ File not found: {args.file}")
                return 1
        else:
            # Interactive file selection
            json_files = selector.get_available_json_files()
            if not json_files:
                print(f"❌ No JSON files found in {DATAQA_FOLDER} folder")
                return 1
            
            if args.non_interactive:
                # Use first file in non-interactive mode
                file_path = json_files[0]
                print(f"📁 Using first available file: {file_path.name}")
            else:
                file_path = selector.display_file_selection(json_files)
        
        # Run evaluation
        output_file = selector.run_evaluation(file_path, args.max_questions)
        
        # Load results and calculate statistics
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = data['results']
        stats = selector.calculate_statistics(results)
        
        # Print results
        selector.print_results(
            stats, 
            file_path.name, 
            data['total_questions'], 
            data['successful'], 
            data['failed']
        )
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📄 Results saved to: {output_file}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n👋 Evaluation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"❌ Evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
