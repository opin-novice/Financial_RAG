#!/usr/bin/env python3
"""
Comprehensive RAG Model Evaluation Script
=========================================
Evaluates both Advanced RAG (main2.py) and Vanilla RAG (main.py) models
using BNqapair.json and calculates cosine similarity scores
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Import RAG systems
try:
    from main2 import CoreRAGSystem as AdvancedRAGSystem
    ADVANCED_RAG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advanced RAG system not available: {e}")
    ADVANCED_RAG_AVAILABLE = False

try:
    from main import VanillaRAGSystem
    VANILLA_RAG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Vanilla RAG system not available: {e}")
    VANILLA_RAG_AVAILABLE = False

# Configuration
EMBEDDING_MODEL = "BAAI/bge-m3"
LOGS_FOLDER = "logs"
RESULTS_FOLDER = "evaluation_results"

# Set up logging
os.makedirs(LOGS_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{LOGS_FOLDER}/rag_evaluation.log', encoding='utf-8'),
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
        self.advanced_rag_response = ""
        self.vanilla_rag_response = ""
        self.advanced_rag_similarity = 0.0
        self.vanilla_rag_similarity = 0.0
        self.advanced_rag_metadata = {}
        self.vanilla_rag_metadata = {}
        self.advanced_rag_processing_time = 0.0
        self.vanilla_rag_processing_time = 0.0
        self.advanced_rag_error = None
        self.vanilla_rag_error = None

class RAGModelEvaluator:
    """Comprehensive RAG model evaluator"""
    
    def __init__(self):
        """Initialize the evaluator"""
        logger.info("🚀 Initializing RAG Model Evaluator...")
        self._init_embedding_model()
        self._init_rag_systems()
        logger.info("✅ RAG Model Evaluator initialized successfully")
    
    def _init_embedding_model(self):
        """Initialize the embedding model"""
        try:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _init_rag_systems(self):
        """Initialize RAG systems"""
        self.advanced_rag = None
        self.vanilla_rag = None
        
        # Initialize Advanced RAG
        if ADVANCED_RAG_AVAILABLE:
            try:
                logger.info("Initializing Advanced RAG system...")
                self.advanced_rag = AdvancedRAGSystem()
                logger.info("✅ Advanced RAG system initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Advanced RAG system: {e}")
                self.advanced_rag = None
        
        # Initialize Vanilla RAG
        if VANILLA_RAG_AVAILABLE:
            try:
                logger.info("Initializing Vanilla RAG system...")
                self.vanilla_rag = VanillaRAGSystem()
                logger.info("✅ Vanilla RAG system initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Vanilla RAG system: {e}")
                self.vanilla_rag = None
        
        if not self.advanced_rag and not self.vanilla_rag:
            raise Exception("No RAG systems available for evaluation")
    
    def load_ground_truth(self, file_path: str) -> List[Dict]:
        """Load ground truth data from JSON file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Ground truth file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"✅ Loaded {len(data)} questions from {os.path.basename(file_path)}")
        return data
    
    def get_advanced_rag_response(self, question: str) -> Tuple[str, Dict]:
        """Get response from Advanced RAG system"""
        if not self.advanced_rag:
            return "", {"error": "Advanced RAG system not available"}
        
        try:
            result = self.advanced_rag.process_query(question)
            response = result.get('response', '')
            
            metadata = {
                'sources': result.get('sources', []),
                'contexts': result.get('contexts', []),
                'detected_language': result.get('detected_language', 'unknown'),
                'language_confidence': result.get('language_confidence', 0.0),
                'retrieval_method': result.get('retrieval_method', 'unknown'),
                'documents_found': result.get('documents_found', 0),
                'documents_used': result.get('documents_used', 0),
                'cross_encoder_used': result.get('cross_encoder_used', False)
            }
            return response, metadata
        except Exception as e:
            logger.error(f"Failed to get Advanced RAG response: {e}")
            return "", {"error": str(e)}
    
    def get_vanilla_rag_response(self, question: str) -> Tuple[str, Dict]:
        """Get response from Vanilla RAG system"""
        if not self.vanilla_rag:
            return "", {"error": "Vanilla RAG system not available"}
        
        try:
            result = self.vanilla_rag.process_query(question)
            response = result.get('response', '')
            
            metadata = {
                'sources': result.get('sources', []),
                'contexts': result.get('contexts', []),
                'original_query': result.get('original_query', ''),
                'retrieval_method': 'vanilla_rag',
                'rag_type': 'vanilla'
            }
            return response, metadata
        except Exception as e:
            logger.error(f"Failed to get Vanilla RAG response: {e}")
            return "", {"error": str(e)}
    
    def compute_embeddings(self, text: str) -> np.ndarray:
        """Compute embeddings for text"""
        if not text or not text.strip():
            return np.array([])
        return self.embedding_model.encode([text.strip()], normalize_embeddings=True, show_progress_bar=False)
    
    def calculate_similarity(self, response: str, ground_truth: str) -> float:
        """Calculate cosine similarity between response and ground truth"""
        if not response or not ground_truth:
            return 0.0
        
        response_embeddings = self.compute_embeddings(response)
        ground_truth_embeddings = self.compute_embeddings(ground_truth)
        
        if response_embeddings.size > 0 and ground_truth_embeddings.size > 0:
            similarity = cosine_similarity(response_embeddings, ground_truth_embeddings)[0][0]
            return float(similarity)
        
        return 0.0
    
    def evaluate_single_question(self, question_data: Dict, question_id: int) -> EvaluationMetrics:
        """Evaluate a single question with both RAG systems"""
        question = question_data['question']
        ground_truth_answer = question_data.get('answer', question_data.get('predicted_answer', ''))
        metrics = EvaluationMetrics(question_id, question, ground_truth_answer)
        
        # Evaluate Advanced RAG
        if self.advanced_rag:
            try:
                start_time = datetime.now()
                response, metadata = self.get_advanced_rag_response(question)
                metrics.advanced_rag_response = response
                metrics.advanced_rag_metadata = metadata
                metrics.advanced_rag_processing_time = (datetime.now() - start_time).total_seconds()
                
                if response and ground_truth_answer:
                    similarity = self.calculate_similarity(response, ground_truth_answer)
                    metrics.advanced_rag_similarity = similarity
                    logger.info(f"Question {question_id}: Advanced RAG Similarity = {similarity:.4f}")
                else:
                    metrics.advanced_rag_error = "No response or ground truth available"
            except Exception as e:
                metrics.advanced_rag_error = str(e)
                logger.error(f"Advanced RAG evaluation failed for question {question_id}: {e}")
        
        # Evaluate Vanilla RAG
        if self.vanilla_rag:
            try:
                start_time = datetime.now()
                response, metadata = self.get_vanilla_rag_response(question)
                metrics.vanilla_rag_response = response
                metrics.vanilla_rag_metadata = metadata
                metrics.vanilla_rag_processing_time = (datetime.now() - start_time).total_seconds()
                
                if response and ground_truth_answer:
                    similarity = self.calculate_similarity(response, ground_truth_answer)
                    metrics.vanilla_rag_similarity = similarity
                    logger.info(f"Question {question_id}: Vanilla RAG Similarity = {similarity:.4f}")
                else:
                    metrics.vanilla_rag_error = "No response or ground truth available"
            except Exception as e:
                metrics.vanilla_rag_error = str(e)
                logger.error(f"Vanilla RAG evaluation failed for question {question_id}: {e}")
        
        return metrics
    
    def run_evaluation(self, ground_truth_file: str, max_questions: int = None) -> str:
        """Run comprehensive evaluation on both RAG systems"""
        print(f"\n🔍 Running comprehensive RAG evaluation on: {os.path.basename(ground_truth_file)}")
        
        ground_truth_data = self.load_ground_truth(ground_truth_file)
        
        if max_questions:
            ground_truth_data = ground_truth_data[:max_questions]
            print(f"📊 Evaluating {len(ground_truth_data)} questions (limited to {max_questions})")
        else:
            print(f"📊 Evaluating all {len(ground_truth_data)} questions")
        
        results = []
        advanced_rag_successful = 0
        advanced_rag_failed = 0
        vanilla_rag_successful = 0
        vanilla_rag_failed = 0
        
        for i, qdata in enumerate(ground_truth_data):
            print(f"\n📝 Processing question {i+1}/{len(ground_truth_data)}: {qdata['question'][:50]}...")
            
            metrics = self.evaluate_single_question(qdata, i)
            results.append({
                'question_id': metrics.question_id,
                'question': metrics.question,
                'ground_truth': metrics.ground_truth,
                'advanced_rag_response': metrics.advanced_rag_response,
                'vanilla_rag_response': metrics.vanilla_rag_response,
                'advanced_rag_similarity': metrics.advanced_rag_similarity,
                'vanilla_rag_similarity': metrics.vanilla_rag_similarity,
                'advanced_rag_metadata': metrics.advanced_rag_metadata,
                'vanilla_rag_metadata': metrics.vanilla_rag_metadata,
                'advanced_rag_processing_time': metrics.advanced_rag_processing_time,
                'vanilla_rag_processing_time': metrics.vanilla_rag_processing_time,
                'advanced_rag_error': metrics.advanced_rag_error,
                'vanilla_rag_error': metrics.vanilla_rag_error
            })
            
            # Count successes and failures
            if metrics.advanced_rag_error:
                advanced_rag_failed += 1
            else:
                advanced_rag_successful += 1
            
            if metrics.vanilla_rag_error:
                vanilla_rag_failed += 1
            else:
                vanilla_rag_successful += 1
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = os.path.splitext(os.path.basename(ground_truth_file))[0]
        output_file = f"{RESULTS_FOLDER}/comprehensive_rag_evaluation_{filename_base}_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'test_file': ground_truth_file,
                'evaluation_timestamp': timestamp,
                'total_questions': len(ground_truth_data),
                'advanced_rag_successful': advanced_rag_successful,
                'advanced_rag_failed': advanced_rag_failed,
                'vanilla_rag_successful': vanilla_rag_successful,
                'vanilla_rag_failed': vanilla_rag_failed,
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Results saved to: {output_file}")
        return output_file
    
    def calculate_statistics(self, results: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for both RAG systems"""
        advanced_rag_similarities = []
        vanilla_rag_similarities = []
        
        for result in results:
            if 'advanced_rag_similarity' in result and result['advanced_rag_similarity'] is not None:
                advanced_rag_similarities.append(float(result['advanced_rag_similarity']))
            
            if 'vanilla_rag_similarity' in result and result['vanilla_rag_similarity'] is not None:
                vanilla_rag_similarities.append(float(result['vanilla_rag_similarity']))
        
        stats = {}
        
        # Advanced RAG statistics
        if advanced_rag_similarities:
            stats['advanced_rag'] = {
                'mean': np.mean(advanced_rag_similarities),
                'median': np.median(advanced_rag_similarities),
                'std': np.std(advanced_rag_similarities),
                'min': np.min(advanced_rag_similarities),
                'max': np.max(advanced_rag_similarities),
                'count': len(advanced_rag_similarities)
            }
        else:
            stats['advanced_rag'] = {}
        
        # Vanilla RAG statistics
        if vanilla_rag_similarities:
            stats['vanilla_rag'] = {
                'mean': np.mean(vanilla_rag_similarities),
                'median': np.median(vanilla_rag_similarities),
                'std': np.std(vanilla_rag_similarities),
                'min': np.min(vanilla_rag_similarities),
                'max': np.max(vanilla_rag_similarities),
                'count': len(vanilla_rag_similarities)
            }
        else:
            stats['vanilla_rag'] = {}
        
        return stats
    
    def print_results(self, stats: Dict[str, Dict[str, float]], file_path: str, 
                     total_questions: int, advanced_rag_successful: int, advanced_rag_failed: int,
                     vanilla_rag_successful: int, vanilla_rag_failed: int):
        """Print formatted results for both RAG systems"""
        print("\n" + "="*80)
        print("📈 COMPREHENSIVE RAG EVALUATION RESULTS")
        print(f"📄 Test File: {os.path.basename(file_path)}")
        print("="*80)
        
        print(f"📊 Total Questions: {total_questions}")
        print()
        
        # Advanced RAG Results
        print("🚀 ADVANCED RAG SYSTEM:")
        print(f"  ✅ Successful: {advanced_rag_successful}")
        print(f"  ❌ Failed: {advanced_rag_failed}")
        
        if stats.get('advanced_rag'):
            adv_stats = stats['advanced_rag']
            print(f"  📈 Mean Similarity:     {adv_stats['mean']:.4f}")
            print(f"  📈 Median Similarity:   {adv_stats['median']:.4f}")
            print(f"  📈 Std Dev:             {adv_stats['std']:.4f}")
            print(f"  📈 Min Similarity:      {adv_stats['min']:.4f}")
            print(f"  📈 Max Similarity:      {adv_stats['max']:.4f}")
            
            # Quality assessment for Advanced RAG
            mean_score = adv_stats['mean']
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
            
            print(f"  🎯 Overall Quality: {quality} (Mean: {mean_score:.4f})")
        else:
            print("  ❌ No valid similarity data to analyze")
        
        print()
        
        # Vanilla RAG Results
        print("🍦 VANILLA RAG SYSTEM:")
        print(f"  ✅ Successful: {vanilla_rag_successful}")
        print(f"  ❌ Failed: {vanilla_rag_failed}")
        
        if stats.get('vanilla_rag'):
            van_stats = stats['vanilla_rag']
            print(f"  📈 Mean Similarity:     {van_stats['mean']:.4f}")
            print(f"  📈 Median Similarity:   {van_stats['median']:.4f}")
            print(f"  📈 Std Dev:             {van_stats['std']:.4f}")
            print(f"  📈 Min Similarity:      {van_stats['min']:.4f}")
            print(f"  📈 Max Similarity:      {van_stats['max']:.4f}")
            
            # Quality assessment for Vanilla RAG
            mean_score = van_stats['mean']
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
            
            print(f"  🎯 Overall Quality: {quality} (Mean: {mean_score:.4f})")
        else:
            print("  ❌ No valid similarity data to analyze")
        
        print()
        
        # Comparison
        if stats.get('advanced_rag') and stats.get('vanilla_rag'):
            adv_mean = stats['advanced_rag']['mean']
            van_mean = stats['vanilla_rag']['mean']
            improvement = ((adv_mean - van_mean) / van_mean) * 100 if van_mean > 0 else 0
            
            print("🆚 COMPARISON:")
            print(f"  Advanced RAG vs Vanilla RAG: {adv_mean:.4f} vs {van_mean:.4f}")
            print(f"  Improvement: {improvement:+.2f}%")
            
            if improvement > 0:
                print("  🏆 Advanced RAG performs better!")
            elif improvement < 0:
                print("  🏆 Vanilla RAG performs better!")
            else:
                print("  🤝 Both systems perform equally!")
        
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Comprehensive RAG Model Evaluation")
    parser.add_argument('--file', '-f', default='BNqapair.json', help='JSON file to test (default: BNqapair.json)')
    parser.add_argument('--max-questions', '-m', type=int, help='Maximum number of questions to test')
    parser.add_argument('--output-dir', '-o', default='evaluation_results', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Update output directory
    global RESULTS_FOLDER
    RESULTS_FOLDER = args.output_dir
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    
    try:
        evaluator = RAGModelEvaluator()
        
        # Check if file exists
        if not os.path.exists(args.file):
            print(f"❌ File not found: {args.file}")
            return 1
        
        # Run evaluation
        output_file = evaluator.run_evaluation(args.file, args.max_questions)
        
        # Load results and calculate statistics
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = data['results']
        stats = evaluator.calculate_statistics(results)
        
        # Print results
        evaluator.print_results(
            stats, 
            args.file, 
            data['total_questions'], 
            data['advanced_rag_successful'], 
            data['advanced_rag_failed'],
            data['vanilla_rag_successful'], 
            data['vanilla_rag_failed']
        )
        
        print(f"\n🎉 Comprehensive evaluation completed successfully!")
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
