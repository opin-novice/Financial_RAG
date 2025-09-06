#!/usr/bin/env python3
"""
Comprehensive BertScore Evaluation for RAG Systems
=================================================
Evaluates both Vanilla and Advanced RAG on BNqapair and ENGqapair datasets
Generates complete BertScore metrics for research paper
"""

import os
import sys
import json
import time
import threading
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from pathlib import Path

# Import RAG systems with error handling
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

# Import BertScore evaluator
from evalbertscore import BERTScoreEvaluator

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

def process_with_timeout(func, timeout_seconds):
    """Execute function with timeout using threading"""
    result = [None]
    error = [None]
    
    def target():
        try:
            result[0] = func()
        except Exception as e:
            error[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
    
    if error[0]:
        raise error[0]
    
    return result[0]

class ComprehensiveRAGEvaluator:
    """Comprehensive RAG evaluator for multiple datasets and systems"""
    
    def __init__(self, timeout_seconds: int = 20):
        """
        Initialize comprehensive evaluator
        
        Args:
            timeout_seconds: Timeout for each question processing
        """
        self.timeout_seconds = timeout_seconds
        self.advanced_rag = None
        self.vanilla_rag = None
        self.bertscore_evaluator = None
        
        # Initialize systems
        self._init_rag_systems()
        self._init_bertscore_evaluator()
    
    def _init_rag_systems(self):
        """Initialize RAG systems"""
        print("[INFO] Initializing RAG systems...")
        
        # Initialize Advanced RAG
        if ADVANCED_RAG_AVAILABLE:
            try:
                print("[INFO] Loading Advanced RAG system...")
                self.advanced_rag = AdvancedRAGSystem()
                print("[INFO] ✅ Advanced RAG system loaded successfully")
            except Exception as e:
                print(f"[ERROR] Failed to load Advanced RAG: {e}")
                self.advanced_rag = None
        
        # Initialize Vanilla RAG
        if VANILLA_RAG_AVAILABLE:
            try:
                print("[INFO] Loading Vanilla RAG system...")
                self.vanilla_rag = VanillaRAGSystem()
                print("[INFO] ✅ Vanilla RAG system loaded successfully")
            except Exception as e:
                print(f"[ERROR] Failed to load Vanilla RAG: {e}")
                self.vanilla_rag = None
        
        if not self.advanced_rag and not self.vanilla_rag:
            raise Exception("No RAG systems available for evaluation")
    
    def _init_bertscore_evaluator(self):
        """Initialize BertScore evaluator"""
        try:
            print("[INFO] Initializing BertScore evaluator...")
            self.bertscore_evaluator = BERTScoreEvaluator(verbose=False)
            print("[INFO] ✅ BertScore evaluator ready")
        except Exception as e:
            print(f"[ERROR] Failed to initialize BertScore: {e}")
            raise e
    
    def process_question_with_timeout(self, question: str, system_name: str) -> Tuple[str, Dict, Optional[str]]:
        """
        Process a single question with timeout
        
        Args:
            question: Question to process
            system_name: Name of the RAG system ('advanced' or 'vanilla')
            
        Returns:
            Tuple of (response, metadata, error_message)
        """
        def process_question():
            if system_name == 'advanced' and self.advanced_rag:
                result = self.advanced_rag.process_query(question)
                response = result.get('response', '')
                metadata = {
                    'sources': result.get('sources', []),
                    'contexts': result.get('contexts', []),
                    'detected_language': result.get('detected_language', 'unknown'),
                    'retrieval_method': result.get('retrieval_method', 'unknown')
                }
                return response, metadata, None
                
            elif system_name == 'vanilla' and self.vanilla_rag:
                result = self.vanilla_rag.process_query(question)
                response = result.get('response', '')
                metadata = {
                    'sources': result.get('sources', []),
                    'contexts': result.get('contexts', []),
                    'retrieval_method': 'vanilla_rag'
                }
                return response, metadata, None
            else:
                return "", {}, f"{system_name} RAG system not available"
        
        try:
            return process_with_timeout(process_question, self.timeout_seconds)
        except TimeoutError as e:
            return "", {}, str(e)
        except Exception as e:
            return "", {}, str(e)
    
    def evaluate_dataset(self, dataset_name: str, questions: List[Dict]) -> Dict:
        """
        Evaluate a dataset with both RAG systems
        
        Args:
            dataset_name: Name of the dataset (e.g., 'BNqapair', 'ENGqapair')
            questions: List of question dictionaries
            
        Returns:
            Dictionary containing evaluation results
        """
        print(f"\n{'='*80}")
        print(f"EVALUATING {dataset_name.upper()} DATASET")
        print(f"Total questions: {len(questions)}")
        print(f"{'='*80}")
        
        results = {
            'dataset_name': dataset_name,
            'total_questions': len(questions),
            'vanilla_rag': {'responses': [], 'references': [], 'successful': 0, 'failed': 0},
            'advanced_rag': {'responses': [], 'references': [], 'successful': 0, 'failed': 0}
        }
        
        # Process each question
        for i, qdata in enumerate(questions):
            question = qdata['question']
            ground_truth = qdata.get('answer', '')
            
            print(f"\n[INFO] Processing question {i+1}/{len(questions)}: {question[:60]}...")
            
            # Process with Vanilla RAG
            if self.vanilla_rag:
                print(f"  [INFO] Processing with Vanilla RAG...")
                response, metadata, error = self.process_question_with_timeout(question, 'vanilla')
                
                if error:
                    print(f"  [ERROR] Vanilla RAG failed: {error}")
                    results['vanilla_rag']['failed'] += 1
                else:
                    print(f"  [INFO] Vanilla RAG completed: {len(response)} chars")
                    results['vanilla_rag']['responses'].append(response)
                    results['vanilla_rag']['references'].append(ground_truth)
                    results['vanilla_rag']['successful'] += 1
            
            # Process with Advanced RAG
            if self.advanced_rag:
                print(f"  [INFO] Processing with Advanced RAG...")
                response, metadata, error = self.process_question_with_timeout(question, 'advanced')
                
                if error:
                    print(f"  [ERROR] Advanced RAG failed: {error}")
                    results['advanced_rag']['failed'] += 1
                else:
                    print(f"  [INFO] Advanced RAG completed: {len(response)} chars")
                    results['advanced_rag']['responses'].append(response)
                    results['advanced_rag']['references'].append(ground_truth)
                    results['advanced_rag']['successful'] += 1
        
        # Calculate BertScore metrics
        print(f"\n[INFO] Calculating BertScore metrics for {dataset_name}...")
        
        # Vanilla RAG BertScore
        if results['vanilla_rag']['responses']:
            print(f"  [INFO] Computing Vanilla RAG BertScore ({len(results['vanilla_rag']['responses'])} responses)...")
            vanilla_scores = self.bertscore_evaluator.evaluate_batch(
                results['vanilla_rag']['responses'],
                results['vanilla_rag']['references']
            )
            results['vanilla_rag']['bertscore'] = vanilla_scores
        else:
            results['vanilla_rag']['bertscore'] = None
        
        # Advanced RAG BertScore
        if results['advanced_rag']['responses']:
            print(f"  [INFO] Computing Advanced RAG BertScore ({len(results['advanced_rag']['responses'])} responses)...")
            advanced_scores = self.bertscore_evaluator.evaluate_batch(
                results['advanced_rag']['responses'],
                results['advanced_rag']['references']
            )
            results['advanced_rag']['bertscore'] = advanced_scores
        else:
            results['advanced_rag']['bertscore'] = None
        
        return results
    
    def run_comprehensive_evaluation(self) -> Dict:
        """
        Run comprehensive evaluation on all datasets
        
        Returns:
            Dictionary containing all evaluation results
        """
        print("="*80)
        print("COMPREHENSIVE BERTSCORE EVALUATION")
        print("="*80)
        
        all_results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'timeout_seconds': self.timeout_seconds,
            'datasets': {}
        }
        
        # Load and evaluate BNqapair dataset
        print("\n[INFO] Loading BNqapair dataset...")
        with open('dataqa/BNqapair.json', 'r', encoding='utf-8') as f:
            bn_questions = json.load(f)
        
        bn_results = self.evaluate_dataset('BNqapair', bn_questions)
        all_results['datasets']['BNqapair'] = bn_results
        
        # Load and evaluate ENGqapair dataset
        print("\n[INFO] Loading ENGqapair dataset...")
        with open('dataqa/ENGqapair.json', 'r', encoding='utf-8') as f:
            eng_questions = json.load(f)
        
        eng_results = self.evaluate_dataset('ENGqapair', eng_questions)
        all_results['datasets']['ENGqapair'] = eng_results
        
        return all_results
    
    def print_research_results(self, results: Dict):
        """Print results formatted for research paper"""
        print("\n" + "="*80)
        print("BERTSCORE EVALUATION RESULTS FOR RESEARCH PAPER")
        print("="*80)
        
        for dataset_name, dataset_results in results['datasets'].items():
            print(f"\n📊 {dataset_name} Dataset Results:")
            print("-" * 50)
            
            # Vanilla RAG results
            vanilla = dataset_results['vanilla_rag']
            if vanilla['bertscore']:
                print(f"Vanilla RAG (n={vanilla['successful']}):")
                print(f"  F1-Score:    {vanilla['bertscore']['bert_score_f1']:.4f}")
                print(f"  Precision:   {vanilla['bertscore']['bert_score_precision']:.4f}")
                print(f"  Recall:      {vanilla['bertscore']['bert_score_recall']:.4f}")
            else:
                print(f"Vanilla RAG: No successful responses")
            
            # Advanced RAG results
            advanced = dataset_results['advanced_rag']
            if advanced['bertscore']:
                print(f"Advanced RAG (n={advanced['successful']}):")
                print(f"  F1-Score:    {advanced['bertscore']['bert_score_f1']:.4f}")
                print(f"  Precision:   {advanced['bertscore']['bert_score_precision']:.4f}")
                print(f"  Recall:      {advanced['bertscore']['bert_score_recall']:.4f}")
            else:
                print(f"Advanced RAG: No successful responses")
            
            # Comparison
            if vanilla['bertscore'] and advanced['bertscore']:
                f1_improvement = ((advanced['bertscore']['bert_score_f1'] - vanilla['bertscore']['bert_score_f1']) / vanilla['bertscore']['bert_score_f1']) * 100
                print(f"F1-Score Improvement: {f1_improvement:+.2f}%")
        
        print("\n" + "="*80)
        print("COPY-PASTE READY FOR RESEARCH PAPER:")
        print("="*80)
        
        for dataset_name, dataset_results in results['datasets'].items():
            vanilla = dataset_results['vanilla_rag']
            advanced = dataset_results['advanced_rag']
            
            print(f"\n{dataset_name}:")
            if vanilla['bertscore']:
                print(f"  Vanilla RAG: F1={vanilla['bertscore']['bert_score_f1']:.4f}, "
                      f"Precision={vanilla['bertscore']['bert_score_precision']:.4f}, "
                      f"Recall={vanilla['bertscore']['bert_score_recall']:.4f}")
            if advanced['bertscore']:
                print(f"  Advanced RAG: F1={advanced['bertscore']['bert_score_f1']:.4f}, "
                      f"Precision={advanced['bertscore']['bert_score_precision']:.4f}, "
                      f"Recall={advanced['bertscore']['bert_score_recall']:.4f}")
        
        print("="*80)

def main():
    """Main function"""
    print("="*80)
    print("COMPREHENSIVE BERTSCORE EVALUATION FOR RAG SYSTEMS")
    print("="*80)
    
    try:
        # Initialize evaluator
        evaluator = ComprehensiveRAGEvaluator(timeout_seconds=20)
        
        # Run comprehensive evaluation
        results = evaluator.run_comprehensive_evaluation()
        
        # Print results
        evaluator.print_research_results(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"comprehensive_bertscore_evaluation_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n[INFO] Evaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
