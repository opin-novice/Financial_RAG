#!/usr/bin/env python3
"""
Simple BERTscore Evaluation for ENGqapairs
==========================================
A simplified approach that processes questions one at a time to avoid threading issues
"""

import os
import sys
import json
import time
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Import RAG systems
try:
    from main import VanillaRAGSystem
    VANILLA_RAG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Vanilla RAG system not available: {e}")
    VANILLA_RAG_AVAILABLE = False

try:
    from main2 import CoreRAGSystem as AdvancedRAGSystem
    ADVANCED_RAG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advanced RAG system not available: {e}")
    ADVANCED_RAG_AVAILABLE = False

# Import BertScore
from bert_score import score

class SimpleENGqapairEvaluator:
    """Simple evaluator that processes questions sequentially"""
    
    def __init__(self, max_questions: int = 5):
        self.max_questions = max_questions
        self.vanilla_rag = None
        self.advanced_rag = None
        
        # Initialize systems
        self._init_systems()
    
    def _init_systems(self):
        """Initialize RAG systems one at a time"""
        print("[INFO] Initializing RAG systems...")
        
        # Initialize Vanilla RAG first
        if VANILLA_RAG_AVAILABLE:
            try:
                print("[INFO] Loading Vanilla RAG system...")
                self.vanilla_rag = VanillaRAGSystem()
                print("[INFO] ✅ Vanilla RAG system loaded successfully")
            except Exception as e:
                print(f"[ERROR] Failed to load Vanilla RAG: {e}")
                self.vanilla_rag = None
        
        # Initialize Advanced RAG
        if ADVANCED_RAG_AVAILABLE:
            try:
                print("[INFO] Loading Advanced RAG system...")
                self.advanced_rag = AdvancedRAGSystem()
                print("[INFO] ✅ Advanced RAG system loaded successfully")
            except Exception as e:
                print(f"[ERROR] Failed to load Advanced RAG: {e}")
                self.advanced_rag = None
        
        if not self.vanilla_rag and not self.advanced_rag:
            raise Exception("No RAG systems available for evaluation")
    
    def process_question_safely(self, question: str, system_name: str) -> Tuple[str, bool]:
        """
        Process a question with basic error handling
        
        Args:
            question: Question to process
            system_name: 'vanilla' or 'advanced'
            
        Returns:
            Tuple of (response, success_flag)
        """
        try:
            if system_name == 'vanilla' and self.vanilla_rag:
                result = self.vanilla_rag.process_query(question)
                response = result.get('response', '')
                return response, True
                
            elif system_name == 'advanced' and self.advanced_rag:
                result = self.advanced_rag.process_query(question)
                response = result.get('response', '')
                return response, True
            else:
                return "", False
                
        except Exception as e:
            print(f"[ERROR] {system_name} RAG failed: {e}")
            return "", False
    
    def evaluate_questions(self) -> Dict:
        """Evaluate questions and compute BERTscore"""
        print("="*80)
        print("SIMPLE BERTSCORE EVALUATION FOR ENGQAPAIRS")
        print("="*80)
        
        # Load questions
        print("[INFO] Loading ENGqapairs dataset...")
        with open('dataqa/ENGqapair.json', 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        if self.max_questions:
            questions = questions[:self.max_questions]
            print(f"[INFO] Processing {len(questions)} questions (limited to {self.max_questions})")
        else:
            print(f"[INFO] Processing all {len(questions)} questions")
        
        # Collect responses
        vanilla_responses = []
        advanced_responses = []
        references = []
        
        for i, qdata in enumerate(questions):
            question = qdata['question']
            ground_truth = qdata.get('answer', '')
            
            print(f"\n[INFO] Processing question {i+1}/{len(questions)}: {question[:50]}...")
            
            # Process with Vanilla RAG
            if self.vanilla_rag:
                print("  [INFO] Processing with Vanilla RAG...")
                response, success = self.process_question_safely(question, 'vanilla')
                if success:
                    vanilla_responses.append(response)
                    references.append(ground_truth)
                    print(f"  [INFO] Vanilla RAG: {len(response)} chars")
                else:
                    print("  [ERROR] Vanilla RAG failed")
            
            # Process with Advanced RAG
            if self.advanced_rag:
                print("  [INFO] Processing with Advanced RAG...")
                response, success = self.process_question_safely(question, 'advanced')
                if success:
                    advanced_responses.append(response)
                    if not self.vanilla_rag:  # Only add reference once
                        references.append(ground_truth)
                    print(f"  [INFO] Advanced RAG: {len(response)} chars")
                else:
                    print("  [ERROR] Advanced RAG failed")
        
        # Compute BERTscore
        results = {
            'total_questions': len(questions),
            'vanilla_rag': {'responses': len(vanilla_responses), 'bertscore': None},
            'advanced_rag': {'responses': len(advanced_responses), 'bertscore': None}
        }
        
        # Vanilla RAG BERTscore
        if vanilla_responses and references:
            print(f"\n[INFO] Computing Vanilla RAG BERTscore ({len(vanilla_responses)} responses)...")
            try:
                P, R, F1 = score(vanilla_responses, references, lang='en', verbose=False)
                results['vanilla_rag']['bertscore'] = {
                    'precision': P.mean().item(),
                    'recall': R.mean().item(),
                    'f1': F1.mean().item()
                }
                print(f"[INFO] Vanilla RAG BERTscore: F1={F1.mean().item():.4f}")
            except Exception as e:
                print(f"[ERROR] Vanilla RAG BERTscore failed: {e}")
        
        # Advanced RAG BERTscore
        if advanced_responses and references:
            print(f"\n[INFO] Computing Advanced RAG BERTscore ({len(advanced_responses)} responses)...")
            try:
                P, R, F1 = score(advanced_responses, references, lang='en', verbose=False)
                results['advanced_rag']['bertscore'] = {
                    'precision': P.mean().item(),
                    'recall': R.mean().item(),
                    'f1': F1.mean().item()
                }
                print(f"[INFO] Advanced RAG BERTscore: F1={F1.mean().item():.4f}")
            except Exception as e:
                print(f"[ERROR] Advanced RAG BERTscore failed: {e}")
        
        return results
    
    def print_results(self, results: Dict):
        """Print formatted results"""
        print("\n" + "="*80)
        print("BERTSCORE EVALUATION RESULTS FOR ENGQAPAIRS")
        print("="*80)
        
        print(f"Total Questions: {results['total_questions']}")
        print(f"Vanilla RAG Responses: {results['vanilla_rag']['responses']}")
        print(f"Advanced RAG Responses: {results['advanced_rag']['responses']}")
        
        # Vanilla RAG results
        if results['vanilla_rag']['bertscore']:
            bs = results['vanilla_rag']['bertscore']
            print(f"\nVanilla RAG BERTscore:")
            print(f"  F1-Score:    {bs['f1']:.4f}")
            print(f"  Precision:   {bs['precision']:.4f}")
            print(f"  Recall:      {bs['recall']:.4f}")
        else:
            print(f"\nVanilla RAG: No valid BERTscore data")
        
        # Advanced RAG results
        if results['advanced_rag']['bertscore']:
            bs = results['advanced_rag']['bertscore']
            print(f"\nAdvanced RAG BERTscore:")
            print(f"  F1-Score:    {bs['f1']:.4f}")
            print(f"  Precision:   {bs['precision']:.4f}")
            print(f"  Recall:      {bs['recall']:.4f}")
        else:
            print(f"\nAdvanced RAG: No valid BERTscore data")
        
        # Comparison
        if (results['vanilla_rag']['bertscore'] and 
            results['advanced_rag']['bertscore']):
            van_f1 = results['vanilla_rag']['bertscore']['f1']
            adv_f1 = results['advanced_rag']['bertscore']['f1']
            improvement = ((adv_f1 - van_f1) / van_f1) * 100
            
            print(f"\nComparison:")
            print(f"  F1-Score Improvement: {improvement:+.2f}%")
            
            if improvement > 0:
                print("  🏆 Advanced RAG performs better!")
            elif improvement < 0:
                print("  🏆 Vanilla RAG performs better!")
            else:
                print("  🤝 Both systems perform equally!")
        
        print("\n" + "="*80)
        print("RESEARCH PAPER FORMAT:")
        print("="*80)
        
        if results['vanilla_rag']['bertscore']:
            bs = results['vanilla_rag']['bertscore']
            print(f"Vanilla RAG: F1={bs['f1']:.4f}, Precision={bs['precision']:.4f}, Recall={bs['recall']:.4f}")
        
        if results['advanced_rag']['bertscore']:
            bs = results['advanced_rag']['bertscore']
            print(f"Advanced RAG: F1={bs['f1']:.4f}, Precision={bs['precision']:.4f}, Recall={bs['recall']:.4f}")
        
        print("="*80)

def main():
    """Main function"""
    print("="*80)
    print("SIMPLE BERTSCORE EVALUATION FOR ENGQAPAIRS")
    print("="*80)
    
    try:
        # Initialize evaluator with limited questions
        evaluator = SimpleENGqapairEvaluator(max_questions=3)
        
        # Run evaluation
        results = evaluator.evaluate_questions()
        
        # Print results
        evaluator.print_results(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"simple_eng_bertscore_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'evaluation_timestamp': timestamp,
                'max_questions': evaluator.max_questions,
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
        
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
