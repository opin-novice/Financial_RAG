#!/usr/bin/env python3
"""
Windows-Compatible Batch RAG Evaluation
======================================
Uses threading for timeout instead of signal (Windows compatible)
"""

import os
import sys
import json
import time
import threading
import logging
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

class WindowsBatchRAGEvaluator:
    """Windows-compatible batch RAG evaluator with timeout handling"""
    
    def __init__(self, timeout_seconds: int = 30, batch_size: int = 5):
        """
        Initialize batch evaluator
        
        Args:
            timeout_seconds: Timeout for each question processing
            batch_size: Number of questions to process in each batch
        """
        self.timeout_seconds = timeout_seconds
        self.batch_size = batch_size
        self.advanced_rag = None
        self.vanilla_rag = None
        
        # Initialize RAG systems
        self._init_rag_systems()
    
    def _init_rag_systems(self):
        """Initialize RAG systems with error handling"""
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
    
    def evaluate_batch(self, questions: List[Dict], start_idx: int) -> List[Dict]:
        """
        Evaluate a batch of questions
        
        Args:
            questions: List of question dictionaries
            start_idx: Starting index for this batch
            
        Returns:
            List of evaluation results
        """
        batch_results = []
        
        for i, qdata in enumerate(questions):
            question_idx = start_idx + i
            question = qdata['question']
            ground_truth = qdata.get('answer', '')
            
            print(f"\n[INFO] Processing question {question_idx + 1}: {question[:50]}...")
            
            result = {
                'question_id': question_idx,
                'question': question,
                'ground_truth': ground_truth,
                'advanced_rag_response': '',
                'vanilla_rag_response': '',
                'advanced_rag_metadata': {},
                'vanilla_rag_metadata': {},
                'advanced_rag_error': None,
                'vanilla_rag_error': None,
                'processing_time': 0.0
            }
            
            start_time = time.time()
            
            # Process with Advanced RAG
            if self.advanced_rag:
                print(f"  [INFO] Processing with Advanced RAG...")
                response, metadata, error = self.process_question_with_timeout(question, 'advanced')
                result['advanced_rag_response'] = response
                result['advanced_rag_metadata'] = metadata
                result['advanced_rag_error'] = error
                
                if error:
                    print(f"  [ERROR] Advanced RAG failed: {error}")
                else:
                    print(f"  [INFO] Advanced RAG completed: {len(response)} chars")
            
            # Process with Vanilla RAG
            if self.vanilla_rag:
                print(f"  [INFO] Processing with Vanilla RAG...")
                response, metadata, error = self.process_question_with_timeout(question, 'vanilla')
                result['vanilla_rag_response'] = response
                result['vanilla_rag_metadata'] = metadata
                result['vanilla_rag_error'] = error
                
                if error:
                    print(f"  [ERROR] Vanilla RAG failed: {error}")
                else:
                    print(f"  [INFO] Vanilla RAG completed: {len(response)} chars")
            
            result['processing_time'] = time.time() - start_time
            batch_results.append(result)
            
            print(f"  [INFO] Question {question_idx + 1} completed in {result['processing_time']:.2f}s")
        
        return batch_results
    
    def run_evaluation(self, questions_file: str, output_file: str = None) -> str:
        """
        Run evaluation on all questions in batches
        
        Args:
            questions_file: Path to questions JSON file
            output_file: Output file path (optional)
            
        Returns:
            Path to output file
        """
        print(f"[INFO] Loading questions from {questions_file}...")
        
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        total_questions = len(questions)
        print(f"[INFO] Total questions to process: {total_questions}")
        print(f"[INFO] Batch size: {self.batch_size}")
        print(f"[INFO] Timeout per question: {self.timeout_seconds}s")
        
        all_results = []
        successful_advanced = 0
        failed_advanced = 0
        successful_vanilla = 0
        failed_vanilla = 0
        
        # Process questions in batches
        for batch_start in range(0, total_questions, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_questions)
            batch_questions = questions[batch_start:batch_end]
            
            print(f"\n{'='*60}")
            print(f"PROCESSING BATCH {batch_start//self.batch_size + 1}")
            print(f"Questions {batch_start + 1} to {batch_end} of {total_questions}")
            print(f"{'='*60}")
            
            try:
                batch_results = self.evaluate_batch(batch_questions, batch_start)
                all_results.extend(batch_results)
                
                # Count successes and failures for this batch
                for result in batch_results:
                    if result['advanced_rag_error']:
                        failed_advanced += 1
                    else:
                        successful_advanced += 1
                    
                    if result['vanilla_rag_error']:
                        failed_vanilla += 1
                    else:
                        successful_vanilla += 1
                
                print(f"[INFO] Batch completed. Advanced RAG: {successful_advanced} success, {failed_advanced} failed")
                print(f"[INFO] Vanilla RAG: {successful_vanilla} success, {failed_vanilla} failed")
                
            except Exception as e:
                print(f"[ERROR] Batch processing failed: {e}")
                continue
        
        # Save results
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"windows_batch_rag_evaluation_{timestamp}.json"
        
        results_data = {
            'source_file': questions_file,
            'evaluation_timestamp': datetime.now().isoformat(),
            'total_questions': total_questions,
            'batch_size': self.batch_size,
            'timeout_seconds': self.timeout_seconds,
            'advanced_rag_successful': successful_advanced,
            'advanced_rag_failed': failed_advanced,
            'vanilla_rag_successful': successful_vanilla,
            'vanilla_rag_failed': failed_vanilla,
            'results': all_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[INFO] Evaluation completed!")
        print(f"[INFO] Results saved to: {output_file}")
        print(f"[INFO] Advanced RAG: {successful_advanced}/{total_questions} successful")
        print(f"[INFO] Vanilla RAG: {successful_vanilla}/{total_questions} successful")
        
        return output_file

def main():
    """Main function"""
    print("="*80)
    print("WINDOWS BATCH RAG EVALUATION")
    print("="*80)
    
    # Configuration
    questions_file = "dataqa/BNqapair.json"
    batch_size = 2  # Very small batch size to avoid hanging
    timeout_seconds = 15  # 15 second timeout per question
    
    if not os.path.exists(questions_file):
        print(f"[ERROR] Questions file not found: {questions_file}")
        return 1
    
    try:
        evaluator = WindowsBatchRAGEvaluator(
            timeout_seconds=timeout_seconds,
            batch_size=batch_size
        )
        
        output_file = evaluator.run_evaluation(questions_file)
        
        print(f"\n🎉 Batch evaluation completed successfully!")
        print(f"📄 Results saved to: {output_file}")
        
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
