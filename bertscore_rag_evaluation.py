#!/usr/bin/env python3
"""
BertScore Evaluation for Vanilla and Advanced RAG Systems
========================================================
Performs separate BertScore evaluation for both RAG systems using existing evaluation results
Generates clean scores suitable for research paper reporting
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Tuple
from bert_score import score
import datetime
from pathlib import Path

class RAGBertScoreEvaluator:
    """BertScore evaluator specifically for RAG systems comparison"""
    
    def __init__(self, model_type: str = "bert-base-uncased", verbose: bool = False):
        """
        Initialize BertScore evaluator
        
        Args:
            model_type: BERT model type for evaluation
            verbose: Whether to show verbose output
        """
        print("[INFO] Initializing RAG BertScore Evaluator...")
        self.model_type = model_type
        self.verbose = verbose
        
        # Test BERTScore availability
        try:
            test_pred = ["This is a test."]
            test_ref = ["This is a test."]
            P, R, F1 = score(test_pred, test_ref, lang='en', verbose=False)
            print("[INFO] [OK] BERTScore is working correctly")
        except Exception as e:
            print(f"[ERROR] BERTScore initialization failed: {e}")
            raise e
    
    def evaluate_rag_system(self, predictions: List[str], references: List[str], 
                           system_name: str) -> Dict[str, float]:
        """
        Evaluate a RAG system using BertScore
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            system_name: Name of the RAG system (for logging)
            
        Returns:
            Dictionary of BertScore metrics
        """
        if not predictions or not references:
            print(f"[WARNING] No valid data for {system_name}")
            return self._empty_scores()
        
        if len(predictions) != len(references):
            print(f"[WARNING] Mismatched data lengths for {system_name}: {len(predictions)} vs {len(references)}")
            min_len = min(len(predictions), len(references))
            predictions = predictions[:min_len]
            references = references[:min_len]
        
        print(f"[INFO] Evaluating {system_name} with {len(predictions)} predictions...")
        
        try:
            # Batch BertScore evaluation for efficiency
            P, R, F1 = score(predictions, references, 
                           model_type=self.model_type,
                           lang='en', 
                           verbose=self.verbose)
            
            # Calculate average scores
            avg_precision = P.mean().item()
            avg_recall = R.mean().item()
            avg_f1 = F1.mean().item()
            
            # Calculate standard deviations
            std_precision = P.std().item()
            std_recall = R.std().item()
            std_f1 = F1.std().item()
            
            # Calculate min/max scores
            min_f1 = F1.min().item()
            max_f1 = F1.max().item()
            
            results = {
                "bert_score_precision": avg_precision,
                "bert_score_recall": avg_recall,
                "bert_score_f1": avg_f1,
                "bert_score_precision_std": std_precision,
                "bert_score_recall_std": std_recall,
                "bert_score_f1_std": std_f1,
                "bert_score_f1_min": min_f1,
                "bert_score_f1_max": max_f1,
                "total_evaluations": len(predictions)
            }
            
            print(f"[INFO] {system_name} BertScore Results:")
            print(f"  Precision: {avg_precision:.4f} ± {std_precision:.4f}")
            print(f"  Recall:    {avg_recall:.4f} ± {std_recall:.4f}")
            print(f"  F1-Score:  {avg_f1:.4f} ± {std_f1:.4f}")
            print(f"  F1 Range:  [{min_f1:.4f}, {max_f1:.4f}]")
            
            return results
            
        except Exception as e:
            print(f"[ERROR] BertScore evaluation failed for {system_name}: {e}")
            return self._empty_scores()
    
    def _empty_scores(self) -> Dict[str, float]:
        """Return empty scores dictionary"""
        return {
            "bert_score_precision": 0.0,
            "bert_score_recall": 0.0,
            "bert_score_f1": 0.0,
            "bert_score_precision_std": 0.0,
            "bert_score_recall_std": 0.0,
            "bert_score_f1_std": 0.0,
            "bert_score_f1_min": 0.0,
            "bert_score_f1_max": 0.0,
            "total_evaluations": 0
        }
    
    def load_evaluation_results(self, results_file: str) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Load evaluation results and extract predictions and references
        
        Args:
            results_file: Path to evaluation results JSON file
            
        Returns:
            Tuple of (vanilla_predictions, vanilla_references, advanced_predictions, advanced_references)
        """
        print(f"[INFO] Loading evaluation results from {results_file}...")
        
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = data.get('results', [])
        if not results:
            raise ValueError("No results found in evaluation file")
        
        vanilla_predictions = []
        vanilla_references = []
        advanced_predictions = []
        advanced_references = []
        
        for result in results:
            # Extract ground truth (reference)
            ground_truth = result.get('ground_truth', '')
            
            # Extract vanilla RAG response
            vanilla_response = result.get('vanilla_rag_response', '')
            if vanilla_response and ground_truth:
                vanilla_predictions.append(vanilla_response)
                vanilla_references.append(ground_truth)
            
            # Extract advanced RAG response
            advanced_response = result.get('advanced_rag_response', '')
            if advanced_response and ground_truth:
                advanced_predictions.append(advanced_response)
                advanced_references.append(ground_truth)
        
        print(f"[INFO] Loaded {len(vanilla_predictions)} vanilla RAG pairs")
        print(f"[INFO] Loaded {len(advanced_predictions)} advanced RAG pairs")
        
        return vanilla_predictions, vanilla_references, advanced_predictions, advanced_references
    
    def evaluate_both_systems(self, results_file: str) -> Dict[str, Dict[str, float]]:
        """
        Evaluate both RAG systems using BertScore
        
        Args:
            results_file: Path to evaluation results JSON file
            
        Returns:
            Dictionary containing scores for both systems
        """
        # Load data
        vanilla_preds, vanilla_refs, advanced_preds, advanced_refs = self.load_evaluation_results(results_file)
        
        # Evaluate Vanilla RAG
        print("\n" + "="*60)
        print("EVALUATING VANILLA RAG SYSTEM")
        print("="*60)
        vanilla_scores = self.evaluate_rag_system(vanilla_preds, vanilla_refs, "Vanilla RAG")
        
        # Evaluate Advanced RAG
        print("\n" + "="*60)
        print("EVALUATING ADVANCED RAG SYSTEM")
        print("="*60)
        advanced_scores = self.evaluate_rag_system(advanced_preds, advanced_refs, "Advanced RAG")
        
        return {
            "vanilla_rag": vanilla_scores,
            "advanced_rag": advanced_scores
        }
    
    def generate_research_paper_scores(self, scores: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """
        Generate clean scores formatted for research paper
        
        Args:
            scores: Dictionary containing scores for both systems
            
        Returns:
            Dictionary of formatted scores for research paper
        """
        research_scores = {}
        
        for system_name, system_scores in scores.items():
            if system_scores["total_evaluations"] > 0:
                # Format scores with proper precision for research paper
                f1_mean = system_scores["bert_score_f1"]
                f1_std = system_scores["bert_score_f1_std"]
                precision_mean = system_scores["bert_score_precision"]
                precision_std = system_scores["bert_score_precision_std"]
                recall_mean = system_scores["bert_score_recall"]
                recall_std = system_scores["bert_score_recall_std"]
                
                # Format as "mean ± std" with appropriate decimal places
                research_scores[f"{system_name}_f1"] = f"{f1_mean:.4f} ± {f1_std:.4f}"
                research_scores[f"{system_name}_precision"] = f"{precision_mean:.4f} ± {precision_std:.4f}"
                research_scores[f"{system_name}_recall"] = f"{recall_mean:.4f} ± {recall_std:.4f}"
                research_scores[f"{system_name}_n"] = str(system_scores["total_evaluations"])
            else:
                research_scores[f"{system_name}_f1"] = "N/A"
                research_scores[f"{system_name}_precision"] = "N/A"
                research_scores[f"{system_name}_recall"] = "N/A"
                research_scores[f"{system_name}_n"] = "0"
        
        return research_scores
    
    def print_research_scores(self, research_scores: Dict[str, str]):
        """Print scores formatted for research paper"""
        print("\n" + "="*80)
        print("BERTSCORE EVALUATION RESULTS FOR RESEARCH PAPER")
        print("="*80)
        
        print("\n📊 BERTScore Metrics (Mean ± Standard Deviation):")
        print("-" * 50)
        
        # Vanilla RAG scores
        print(f"Vanilla RAG System (n={research_scores.get('vanilla_rag_n', 'N/A')}):")
        print(f"  F1-Score:    {research_scores.get('vanilla_rag_f1', 'N/A')}")
        print(f"  Precision:   {research_scores.get('vanilla_rag_precision', 'N/A')}")
        print(f"  Recall:      {research_scores.get('vanilla_rag_recall', 'N/A')}")
        
        print()
        
        # Advanced RAG scores
        print(f"Advanced RAG System (n={research_scores.get('advanced_rag_n', 'N/A')}):")
        print(f"  F1-Score:    {research_scores.get('advanced_rag_f1', 'N/A')}")
        print(f"  Precision:   {research_scores.get('advanced_rag_precision', 'N/A')}")
        print(f"  Recall:      {research_scores.get('advanced_rag_recall', 'N/A')}")
        
        print("\n" + "="*80)
        print("COPY-PASTE READY FOR RESEARCH PAPER:")
        print("="*80)
        print(f"Vanilla RAG: F1={research_scores.get('vanilla_rag_f1', 'N/A')}, "
              f"Precision={research_scores.get('vanilla_rag_precision', 'N/A')}, "
              f"Recall={research_scores.get('vanilla_rag_recall', 'N/A')}")
        print(f"Advanced RAG: F1={research_scores.get('advanced_rag_f1', 'N/A')}, "
              f"Precision={research_scores.get('advanced_rag_precision', 'N/A')}, "
              f"Recall={research_scores.get('advanced_rag_recall', 'N/A')}")
        print("="*80)

def find_latest_evaluation_file() -> str:
    """Find the most recent evaluation results file"""
    results_dir = "evaluation_results"
    if not os.path.exists(results_dir):
        raise FileNotFoundError("Evaluation results directory not found")
    
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json') and 'comprehensive_rag_evaluation' in f]
    if not json_files:
        raise FileNotFoundError("No comprehensive RAG evaluation files found")
    
    # Sort by modification time and get the latest
    json_files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
    latest_file = os.path.join(results_dir, json_files[0])
    
    print(f"[INFO] Using latest evaluation file: {latest_file}")
    return latest_file

def main():
    """Main function for RAG BertScore evaluation"""
    print("="*80)
    print("RAG BERTSCORE EVALUATION FOR RESEARCH PAPER")
    print("="*80)
    
    try:
        # Find latest evaluation results
        results_file = find_latest_evaluation_file()
        
        # Initialize evaluator
        evaluator = RAGBertScoreEvaluator(verbose=True)
        
        # Evaluate both systems
        scores = evaluator.evaluate_both_systems(results_file)
        
        # Generate research paper scores
        research_scores = evaluator.generate_research_paper_scores(scores)
        
        # Print results
        evaluator.print_research_scores(research_scores)
        
        # Save detailed results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"bertscore_rag_evaluation_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "evaluation_timestamp": timestamp,
                "source_file": results_file,
                "detailed_scores": scores,
                "research_paper_scores": research_scores,
                "model_type": evaluator.model_type
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
