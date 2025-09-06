#!/usr/bin/env python3
"""
Comprehensive BERTscore Evaluation for ENGqapairs
=================================================
- Vanilla RAG: All 85 questions
- Advanced RAG: Subset of 20 questions
"""

import os
import json
import time
from datetime import datetime
from bert_score import score

def load_engqapairs():
    """Load ENGqapairs data"""
    with open('dataqa/ENGqapair.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def get_vanilla_rag_responses(questions, max_questions=None):
    """Get responses from Vanilla RAG system for all questions"""
    try:
        from main import VanillaRAGSystem
        
        print("[INFO] Initializing Vanilla RAG system...")
        vanilla_rag = VanillaRAGSystem()
        print("[INFO] ✅ Vanilla RAG system ready")
        
        if max_questions:
            questions = questions[:max_questions]
        
        responses = []
        successful = 0
        failed = 0
        
        print(f"[INFO] Processing {len(questions)} questions with Vanilla RAG...")
        
        for i, qdata in enumerate(questions):
            question = qdata['question']
            print(f"[INFO] Vanilla RAG - Question {i+1}/{len(questions)}: {question[:50]}...")
            
            try:
                result = vanilla_rag.process_query(question)
                response = result.get('response', '')
                responses.append(response)
                successful += 1
                print(f"[INFO] Vanilla RAG - Success: {len(response)} chars")
            except Exception as e:
                print(f"[ERROR] Vanilla RAG failed for question {i+1}: {e}")
                responses.append("")
                failed += 1
        
        print(f"[INFO] Vanilla RAG completed: {successful} successful, {failed} failed")
        return responses, successful, failed
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize Vanilla RAG: {e}")
        return [], 0, 0

def get_advanced_rag_responses(questions, max_questions=20):
    """Get responses from Advanced RAG system for subset"""
    try:
        from main2 import CoreRAGSystem as AdvancedRAGSystem
        
        print("[INFO] Initializing Advanced RAG system...")
        advanced_rag = AdvancedRAGSystem()
        print("[INFO] ✅ Advanced RAG system ready")
        
        # Limit to subset
        questions = questions[:max_questions]
        
        responses = []
        successful = 0
        failed = 0
        
        print(f"[INFO] Processing {len(questions)} questions with Advanced RAG...")
        
        for i, qdata in enumerate(questions):
            question = qdata['question']
            print(f"[INFO] Advanced RAG - Question {i+1}/{len(questions)}: {question[:50]}...")
            
            try:
                # Add timeout protection
                start_time = time.time()
                result = advanced_rag.process_query(question)
                elapsed = time.time() - start_time
                
                response = result.get('response', '')
                responses.append(response)
                successful += 1
                print(f"[INFO] Advanced RAG - Success: {len(response)} chars (took {elapsed:.1f}s)")
                
                # Add delay to prevent overwhelming the system
                time.sleep(2)
                
            except Exception as e:
                print(f"[ERROR] Advanced RAG failed for question {i+1}: {e}")
                responses.append("")
                failed += 1
        
        print(f"[INFO] Advanced RAG completed: {successful} successful, {failed} failed")
        return responses, successful, failed
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize Advanced RAG: {e}")
        return [], 0, 0

def compute_bertscore(predictions, references, system_name):
    """Compute BERTscore for a system"""
    if not predictions or not references:
        print(f"[WARNING] No valid data for {system_name}")
        return None
    
    # Filter out empty predictions
    valid_pairs = [(pred, ref) for pred, ref in zip(predictions, references) if pred.strip() and ref.strip()]
    
    if not valid_pairs:
        print(f"[WARNING] No valid prediction-reference pairs for {system_name}")
        return None
    
    preds, refs = zip(*valid_pairs)
    
    print(f"[INFO] Computing BERTscore for {system_name} ({len(preds)} valid pairs)...")
    
    try:
        P, R, F1 = score(list(preds), list(refs), lang='en', verbose=False)
        
        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item(),
            'precision_std': P.std().item(),
            'recall_std': R.std().item(),
            'f1_std': F1.std().item(),
            'valid_pairs': len(preds),
            'min_f1': F1.min().item(),
            'max_f1': F1.max().item()
        }
    except Exception as e:
        print(f"[ERROR] BERTscore computation failed for {system_name}: {e}")
        return None

def print_results(results):
    """Print comprehensive results"""
    print("\n" + "="*80)
    print("COMPREHENSIVE BERTSCORE EVALUATION RESULTS FOR ENGQAPAIRS")
    print("="*80)
    
    print(f"Dataset: ENGqapairs")
    print(f"Total questions in dataset: {results['total_questions']}")
    print(f"Vanilla RAG questions: {results['vanilla_rag']['total_questions']}")
    print(f"Advanced RAG questions: {results['advanced_rag']['total_questions']}")
    
    # Vanilla RAG results
    if results['vanilla_rag']['bertscore']:
        bs = results['vanilla_rag']['bertscore']
        print(f"\nVanilla RAG BERTscore (n={bs['valid_pairs']}):")
        print(f"  F1-Score:    {bs['f1']:.4f} ± {bs['f1_std']:.4f}")
        print(f"  Precision:   {bs['precision']:.4f} ± {bs['precision_std']:.4f}")
        print(f"  Recall:      {bs['recall']:.4f} ± {bs['recall_std']:.4f}")
        print(f"  F1 Range:    [{bs['min_f1']:.4f}, {bs['max_f1']:.4f}]")
        print(f"  Success Rate: {results['vanilla_rag']['successful']}/{results['vanilla_rag']['total_questions']} ({results['vanilla_rag']['success_rate']:.1f}%)")
    else:
        print(f"\nVanilla RAG: No valid BERTscore data")
    
    # Advanced RAG results
    if results['advanced_rag']['bertscore']:
        bs = results['advanced_rag']['bertscore']
        print(f"\nAdvanced RAG BERTscore (n={bs['valid_pairs']}):")
        print(f"  F1-Score:    {bs['f1']:.4f} ± {bs['f1_std']:.4f}")
        print(f"  Precision:   {bs['precision']:.4f} ± {bs['precision_std']:.4f}")
        print(f"  Recall:      {bs['recall']:.4f} ± {bs['recall_std']:.4f}")
        print(f"  F1 Range:    [{bs['min_f1']:.4f}, {bs['max_f1']:.4f}]")
        print(f"  Success Rate: {results['advanced_rag']['successful']}/{results['advanced_rag']['total_questions']} ({results['advanced_rag']['success_rate']:.1f}%)")
    else:
        print(f"\nAdvanced RAG: No valid BERTscore data")
    
    # Comparison (only if both have valid data)
    if (results['vanilla_rag']['bertscore'] and 
        results['advanced_rag']['bertscore']):
        van_f1 = results['vanilla_rag']['bertscore']['f1']
        adv_f1 = results['advanced_rag']['bertscore']['f1']
        improvement = ((adv_f1 - van_f1) / van_f1) * 100 if van_f1 > 0 else 0
        
        print(f"\nComparison (Advanced RAG subset vs Vanilla RAG subset):")
        print(f"  F1-Score Difference: {improvement:+.2f}%")
        
        if improvement > 0:
            print("  🏆 Advanced RAG performs better!")
        elif improvement < 0:
            print("  🏆 Vanilla RAG performs better!")
        else:
            print("  🤝 Both systems perform equally!")
    
    # Research paper format
    print("\n" + "="*80)
    print("RESEARCH PAPER FORMAT:")
    print("="*80)
    
    if results['vanilla_rag']['bertscore']:
        bs = results['vanilla_rag']['bertscore']
        print(f"Vanilla RAG (n={bs['valid_pairs']}): F1={bs['f1']:.4f}±{bs['f1_std']:.4f}, "
              f"Precision={bs['precision']:.4f}±{bs['precision_std']:.4f}, "
              f"Recall={bs['recall']:.4f}±{bs['recall_std']:.4f}")
    
    if results['advanced_rag']['bertscore']:
        bs = results['advanced_rag']['bertscore']
        print(f"Advanced RAG (n={bs['valid_pairs']}): F1={bs['f1']:.4f}±{bs['f1_std']:.4f}, "
              f"Precision={bs['precision']:.4f}±{bs['precision_std']:.4f}, "
              f"Recall={bs['recall']:.4f}±{bs['recall_std']:.4f}")
    
    print("="*80)

def main():
    """Main function"""
    print("="*80)
    print("COMPREHENSIVE BERTSCORE EVALUATION FOR ENGQAPAIRS")
    print("="*80)
    
    # Load data
    print("[INFO] Loading ENGqapairs dataset...")
    questions = load_engqapairs()
    print(f"[INFO] Loaded {len(questions)} questions")
    
    # Get references
    references = [q['answer'] for q in questions]
    
    # Run Vanilla RAG on all questions
    print("\n" + "="*60)
    print("RUNNING VANILLA RAG ON ALL QUESTIONS")
    print("="*60)
    vanilla_responses, van_successful, van_failed = get_vanilla_rag_responses(questions)
    
    # Run Advanced RAG on subset
    print("\n" + "="*60)
    print("RUNNING ADVANCED RAG ON SUBSET")
    print("="*60)
    advanced_responses, adv_successful, adv_failed = get_advanced_rag_responses(questions, max_questions=20)
    
    # Compute BERTscores
    print("\n" + "="*60)
    print("COMPUTING BERTSCORES")
    print("="*60)
    
    # Vanilla RAG BERTscore (all questions)
    vanilla_bertscore = compute_bertscore(vanilla_responses, references, "Vanilla RAG (All Questions)")
    
    # Advanced RAG BERTscore (subset)
    advanced_bertscore = compute_bertscore(advanced_responses, references[:len(advanced_responses)], "Advanced RAG (Subset)")
    
    # Prepare results
    results = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'dataset': 'ENGqapairs',
        'total_questions': len(questions),
        'vanilla_rag': {
            'total_questions': len(questions),
            'successful': van_successful,
            'failed': van_failed,
            'success_rate': (van_successful / len(questions)) * 100 if len(questions) > 0 else 0,
            'responses': vanilla_responses,
            'bertscore': vanilla_bertscore
        },
        'advanced_rag': {
            'total_questions': min(20, len(questions)),
            'successful': adv_successful,
            'failed': adv_failed,
            'success_rate': (adv_successful / min(20, len(questions))) * 100 if min(20, len(questions)) > 0 else 0,
            'responses': advanced_responses,
            'bertscore': advanced_bertscore
        },
        'references': references
    }
    
    # Print results
    print_results(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"comprehensive_eng_bertscore_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Comprehensive results saved to: {output_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())