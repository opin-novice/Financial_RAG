#!/usr/bin/env python3
"""
Simple BertScore Evaluation for RAG Systems
===========================================
Uses existing evalbertscore.py infrastructure to evaluate RAG systems
"""

import os
import json
import sys
from pathlib import Path

# Add current directory to path to import evalbertscore
sys.path.append('.')

from evalbertscore import BERTScoreEvaluator

def load_rag_data(results_file: str):
    """Load RAG evaluation results and prepare for BertScore evaluation"""
    print(f"[INFO] Loading RAG evaluation results from {results_file}...")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    if not results:
        raise ValueError("No results found in evaluation file")
    
    # Prepare data for BertScore evaluation
    vanilla_data = []
    advanced_data = []
    
    for result in results:
        ground_truth = result.get('ground_truth', '')
        
        # Vanilla RAG data
        vanilla_response = result.get('vanilla_rag_response', '')
        if vanilla_response and ground_truth:
            vanilla_data.append({
                "predicted_answer": vanilla_response,
                "reference_answer": ground_truth,
                "question": result.get('question', '')
            })
        
        # Advanced RAG data
        advanced_response = result.get('advanced_rag_response', '')
        if advanced_response and ground_truth:
            advanced_data.append({
                "predicted_answer": advanced_response,
                "reference_answer": ground_truth,
                "question": result.get('question', '')
            })
    
    print(f"[INFO] Prepared {len(vanilla_data)} vanilla RAG pairs")
    print(f"[INFO] Prepared {len(advanced_data)} advanced RAG pairs")
    
    return vanilla_data, advanced_data

def evaluate_rag_system(data, system_name, evaluator):
    """Evaluate a RAG system using BertScore"""
    if not data:
        print(f"[WARNING] No data available for {system_name}")
        return None
    
    print(f"\n{'='*60}")
    print(f"EVALUATING {system_name.upper()} RAG SYSTEM")
    print(f"{'='*60}")
    
    # Extract predictions and references
    predictions = [item["predicted_answer"] for item in data]
    references = [item["reference_answer"] for item in data]
    
    # Evaluate using batch method
    scores = evaluator.evaluate_batch(predictions, references)
    
    print(f"\n{system_name} BertScore Results:")
    print(f"  BERTScore Precision: {scores['bert_score_precision']:.4f}")
    print(f"  BERTScore Recall:    {scores['bert_score_recall']:.4f}")
    print(f"  BERTScore F1:        {scores['bert_score_f1']:.4f}")
    print(f"  Total Evaluations:   {len(predictions)}")
    
    return scores

def main():
    """Main evaluation function"""
    print("="*80)
    print("SIMPLE BERTSCORE EVALUATION FOR RAG SYSTEMS")
    print("="*80)
    
    # Find latest evaluation results
    results_dir = "evaluation_results"
    if not os.path.exists(results_dir):
        print(f"[ERROR] Results directory not found: {results_dir}")
        return 1
    
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json') and 'comprehensive_rag_evaluation' in f]
    if not json_files:
        print("[ERROR] No comprehensive RAG evaluation files found")
        return 1
    
    # Use the latest file
    json_files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
    latest_file = os.path.join(results_dir, json_files[0])
    
    print(f"[INFO] Using evaluation file: {latest_file}")
    
    try:
        # Load data
        vanilla_data, advanced_data = load_rag_data(latest_file)
        
        # Initialize evaluator
        print("\n[INFO] Initializing BertScore evaluator...")
        evaluator = BERTScoreEvaluator(verbose=True)
        
        # Evaluate Vanilla RAG
        vanilla_scores = evaluate_rag_system(vanilla_data, "Vanilla", evaluator)
        
        # Evaluate Advanced RAG
        advanced_scores = evaluate_rag_system(advanced_data, "Advanced", evaluator)
        
        # Print summary for research paper
        print("\n" + "="*80)
        print("BERTSCORE EVALUATION SUMMARY FOR RESEARCH PAPER")
        print("="*80)
        
        if vanilla_scores:
            print(f"\nVanilla RAG System:")
            print(f"  F1-Score:    {vanilla_scores['bert_score_f1']:.4f}")
            print(f"  Precision:   {vanilla_scores['bert_score_precision']:.4f}")
            print(f"  Recall:      {vanilla_scores['bert_score_recall']:.4f}")
        
        if advanced_scores:
            print(f"\nAdvanced RAG System:")
            print(f"  F1-Score:    {advanced_scores['bert_score_f1']:.4f}")
            print(f"  Precision:   {advanced_scores['bert_score_precision']:.4f}")
            print(f"  Recall:      {advanced_scores['bert_score_recall']:.4f}")
        
        # Comparison
        if vanilla_scores and advanced_scores:
            print(f"\nComparison:")
            f1_improvement = ((advanced_scores['bert_score_f1'] - vanilla_scores['bert_score_f1']) / vanilla_scores['bert_score_f1']) * 100
            print(f"  F1-Score Improvement: {f1_improvement:+.2f}%")
            
            if f1_improvement > 0:
                print("  🏆 Advanced RAG performs better!")
            elif f1_improvement < 0:
                print("  🏆 Vanilla RAG performs better!")
            else:
                print("  🤝 Both systems perform equally!")
        
        print("\n" + "="*80)
        print("COPY-PASTE READY FOR RESEARCH PAPER:")
        print("="*80)
        if vanilla_scores:
            print(f"Vanilla RAG: F1={vanilla_scores['bert_score_f1']:.4f}, "
                  f"Precision={vanilla_scores['bert_score_precision']:.4f}, "
                  f"Recall={vanilla_scores['bert_score_recall']:.4f}")
        if advanced_scores:
            print(f"Advanced RAG: F1={advanced_scores['bert_score_f1']:.4f}, "
                  f"Precision={advanced_scores['bert_score_precision']:.4f}, "
                  f"Recall={advanced_scores['bert_score_recall']:.4f}")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
