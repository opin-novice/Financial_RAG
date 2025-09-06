#!/usr/bin/env python3
"""
Cosine Similarity Evaluation for Vanilla/Advanced RAG Systems
Evaluates the quality of retrieved passages against ground truth answers
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Import your RAG systems
from main import VanillaRAGSystem
# from advanced_rag import AdvancedRAGSystem  # Uncomment if you have Advanced RAG

# Configuration
EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_GROUND_TRUTH_FILE = "dataqa/ENGqapair.json"
MAX_QUESTIONS = None  # Set to None for all questions, or a number to limit

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/cosev3_evaluation.log', encoding='utf-8'),
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

class CosineSimilarityEvaluator:
    """Evaluates cosine similarity between retrieved passages and ground truth answers"""
    
    def __init__(self, ground_truth_file: str = DEFAULT_GROUND_TRUTH_FILE, rag_type: str = "vanilla"):
        """Initialize the evaluator"""
        logger.info("🚀 Initializing Cosine Similarity Evaluator...")
        logger.info(f"📄 Ground truth file: {ground_truth_file}")
        
        self.ground_truth_file = ground_truth_file
        self.rag_type = rag_type
        self._init_embedding_model()
        self._init_rag_system()
        logger.info("✅ Cosine Similarity Evaluator initialized successfully")
    
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
            logger.info(f"Initializing {self.rag_type} RAG system...")
            if self.rag_type == "vanilla":
                self.rag_system = VanillaRAGSystem()
            # elif self.rag_type == "advanced":
            #     self.rag_system = AdvancedRAGSystem()
            else:
                raise ValueError(f"Unknown RAG type: {self.rag_type}")

            # Simple check: process_query must exist
            if not hasattr(self.rag_system, 'process_query') or not callable(self.rag_system.process_query):
                raise Exception("RAG system does not have process_query method")

            logger.info("✅ RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise

    def load_ground_truth(self, file_path: str) -> List[Dict]:
        """Load ground truth data from JSON file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Ground truth file not found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"✅ Loaded {len(data)} questions from ground truth")
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
        if not text or not text.strip():
            return np.array([])
        return self.embedding_model.encode([text.strip()], normalize_embeddings=True, show_progress_bar=False)

    def evaluate_single_question(self, question_data: Dict, question_id: int) -> EvaluationMetrics:
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

    def run_evaluation(self) -> str:
        ground_truth_data = self.load_ground_truth(self.ground_truth_file)
        if MAX_QUESTIONS:
            ground_truth_data = ground_truth_data[:MAX_QUESTIONS]
        results = []
        for i, qdata in enumerate(ground_truth_data):
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
        # Save results
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"logs/rag_cosine_similarity_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({'results': results}, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Results saved to: {output_file}")
        return output_file

def main():
    parser = argparse.ArgumentParser(description="Cosine Similarity Evaluator for RAG")
    parser.add_argument('file', nargs='?', help='Ground truth JSON file')
    parser.add_argument('--interactive', '-i', action='store_true')
    parser.add_argument('--list', '-l', action='store_true')
    parser.add_argument('--max-questions', '-m', type=int)
    parser.add_argument('--rag-type', '-r', default='vanilla', choices=['vanilla', 'advanced'],
                        help='Choose RAG system type')
    args = parser.parse_args()

    if args.max_questions:
        global MAX_QUESTIONS
        MAX_QUESTIONS = args.max_questions

    ground_truth_file = args.file or DEFAULT_GROUND_TRUTH_FILE
    evaluator = CosineSimilarityEvaluator(ground_truth_file, rag_type=args.rag_type)
    output_file = evaluator.run_evaluation()
    print(f"🎉 Evaluation completed. Results saved to: {output_file}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
