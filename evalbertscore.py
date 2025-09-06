#!/usr/bin/env python3
"""
BERTScore Evaluation Script for RAG Pipeline
Focused evaluation using BERTScore metrics for semantic similarity
Works with any JSON file from the dataqa folder
"""
import os
import json
import torch
import numpy as np
import glob
from typing import List, Dict, Tuple, Optional
from bert_score import score
import datetime

class BERTScoreEvaluator:
    """BERTScore-based evaluation for RAG systems"""
    
    def __init__(self, model_type: str = None, num_layers: int = None, verbose: bool = False):
        """
        Initialize BERTScore evaluator
        
        Args:
            model_type: BERT model type (e.g., 'bert-base-uncased', 'roberta-base')
            num_layers: Number of layers to use for evaluation
            verbose: Whether to show verbose output
        """
        print("[INFO] Initializing BERTScore Evaluator...")
        
        self.model_type = model_type
        self.num_layers = num_layers
        self.verbose = verbose
        
        # Test BERTScore availability
        try:
            # Test with dummy data
            test_pred = ["This is a test."]
            test_ref = ["This is a test."]
            P, R, F1 = score(test_pred, test_ref, lang='en', verbose=False)
            print("[INFO] [OK] BERTScore is working correctly")
        except Exception as e:
            print(f"[ERROR] BERTScore initialization failed: {e}")
            raise e
    
    def evaluate_answer(self, predicted: str, reference: str, context: List[str] = None) -> Dict[str, float]:
        """
        Evaluate answer using BERTScore metrics
        
        Args:
            predicted: Predicted answer from RAG system
            reference: Reference/gold standard answer
            context: Retrieved context documents (optional)
            
        Returns:
            Dictionary of BERTScore evaluation scores
        """
        if not predicted or not reference:
            return self._empty_scores()
        
        results = {}
        
        try:
            # Main BERTScore evaluation
            P, R, F1 = score([predicted], [reference], 
                           model_type=self.model_type,
                           num_layers=self.num_layers,
                           lang='en', 
                           verbose=self.verbose)
            
            results.update({
                "bert_score_precision": P.mean().item(),
                "bert_score_recall": R.mean().item(),
                "bert_score_f1": F1.mean().item()
            })
            
        except Exception as e:
            print(f"[WARNING] BERTScore evaluation failed: {e}")
            results.update({
                "bert_score_precision": 0.0,
                "bert_score_recall": 0.0,
                "bert_score_f1": 0.0
            })
        
        # Context-based BERTScore evaluation (if context provided)
        if context:
            try:
                context_scores = self._evaluate_context_bert_scores(predicted, reference, context)
                results.update(context_scores)
            except Exception as e:
                print(f"[WARNING] Context BERTScore evaluation failed: {e}")
                results.update({
                    "bert_answer_context_f1": 0.0,
                    "bert_reference_context_f1": 0.0,
                    "bert_context_consistency": 0.0
                })
        else:
            results.update({
                "bert_answer_context_f1": 0.0,
                "bert_reference_context_f1": 0.0,
                "bert_context_consistency": 0.0
            })
        
        # Calculate composite BERTScore
        results["composite_bert_score"] = self._calculate_composite_score(results)
        
        return results
    
    def _empty_scores(self) -> Dict[str, float]:
        """Return empty scores dictionary"""
        return {
            "bert_score_precision": 0.0,
            "bert_score_recall": 0.0,
            "bert_score_f1": 0.0,
            "bert_answer_context_f1": 0.0,
            "bert_reference_context_f1": 0.0,
            "bert_context_consistency": 0.0,
            "composite_bert_score": 0.0
        }
    
    def _evaluate_context_bert_scores(self, predicted: str, reference: str, context: List[str]) -> Dict[str, float]:
        """
        Evaluate context relevance using BERTScore
        
        Args:
            predicted: Generated answer
            reference: Reference answer
            context: Retrieved context documents
            
        Returns:
            Context-based BERTScore metrics
        """
        if not context:
            return {
                "bert_answer_context_f1": 0.0,
                "bert_reference_context_f1": 0.0,
                "bert_context_consistency": 0.0
            }
        
        try:
            # Answer to context BERTScore
            answer_context_scores = []
            for ctx in context:
                if ctx.strip():  # Skip empty context
                    try:
                        _, _, F1 = score([predicted], [ctx], 
                                       model_type=self.model_type,
                                       num_layers=self.num_layers,
                                       lang='en', 
                                       verbose=False)
                        answer_context_scores.append(F1.item())
                    except Exception as e:
                        print(f"[WARNING] Failed to score answer against context: {e}")
                        answer_context_scores.append(0.0)
            
            bert_answer_context_f1 = float(np.mean(answer_context_scores)) if answer_context_scores else 0.0
            
            # Reference to context BERTScore
            reference_context_scores = []
            for ctx in context:
                if ctx.strip():  # Skip empty context
                    try:
                        _, _, F1 = score([reference], [ctx], 
                                       model_type=self.model_type,
                                       num_layers=self.num_layers,
                                       lang='en', 
                                       verbose=False)
                        reference_context_scores.append(F1.item())
                    except Exception as e:
                        print(f"[WARNING] Failed to score reference against context: {e}")
                        reference_context_scores.append(0.0)
            
            bert_reference_context_f1 = float(np.mean(reference_context_scores)) if reference_context_scores else 0.0
            
            # Context consistency (pairwise BERTScore between context documents)
            if len(context) > 1:
                consistency_scores = []
                valid_contexts = [ctx for ctx in context if ctx.strip()]
                
                for i in range(len(valid_contexts)):
                    for j in range(i+1, len(valid_contexts)):
                        try:
                            _, _, F1 = score([valid_contexts[i]], [valid_contexts[j]], 
                                           model_type=self.model_type,
                                           num_layers=self.num_layers,
                                           lang='en', 
                                           verbose=False)
                            consistency_scores.append(F1.item())
                        except Exception as e:
                            print(f"[WARNING] Failed to score context consistency: {e}")
                            consistency_scores.append(0.0)
                
                bert_context_consistency = float(np.mean(consistency_scores)) if consistency_scores else 0.0
            else:
                bert_context_consistency = 1.0  # Single document is perfectly consistent
            
            return {
                "bert_answer_context_f1": bert_answer_context_f1,
                "bert_reference_context_f1": bert_reference_context_f1,
                "bert_context_consistency": bert_context_consistency
            }
            
        except Exception as e:
            print(f"[WARNING] Context BERTScore calculation failed: {e}")
            return {
                "bert_answer_context_f1": 0.0,
                "bert_reference_context_f1": 0.0,
                "bert_context_consistency": 0.0
            }
    
    def _calculate_composite_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate composite BERTScore
        
        Args:
            scores: Dictionary of individual scores
            
        Returns:
            Composite score (0-1)
        """
        # Weighted combination of BERTScore metrics
        weights = {
            "bert_score_f1": 0.5,              # Most important: main answer quality
            "bert_answer_context_f1": 0.25,    # Answer relevance to context
            "bert_reference_context_f1": 0.15, # Reference quality check
            "bert_context_consistency": 0.1    # Context quality
        }
        
        composite = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in scores:
                # Ensure scores are in valid range (0-1)
                score_value = max(0.0, min(1.0, scores[metric]))
                composite += score_value * weight
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            composite = composite / total_weight
        
        # Ensure final score is in 0-1 range
        composite = max(0.0, min(1.0, composite))
        
        return composite
    
    def evaluate_batch(self, predictions: List[str], references: List[str], 
                      contexts: List[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate batch of predictions using BERTScore
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            contexts: List of retrieved contexts (optional)
            
        Returns:
            Average BERTScore metrics across all predictions
        """
        if not predictions or not references or len(predictions) != len(references):
            return self._empty_scores()
        
        if contexts and len(contexts) != len(predictions):
            contexts = None  # Disable context evaluation if mismatched
        
        # Batch BERTScore evaluation for efficiency
        try:
            print(f"[INFO] Computing batch BERTScore for {len(predictions)} predictions...")
            P, R, F1 = score(predictions, references, 
                           model_type=self.model_type,
                           num_layers=self.num_layers,
                           lang='en', 
                           verbose=self.verbose)
            
            # Calculate average scores
            avg_precision = P.mean().item()
            avg_recall = R.mean().item()
            avg_f1 = F1.mean().item()
            
        except Exception as e:
            print(f"[WARNING] Batch BERTScore evaluation failed: {e}")
            print("[INFO] Falling back to individual evaluation...")
            
            # Fallback to individual evaluation
            total_scores = {"bert_score_precision": 0.0, "bert_score_recall": 0.0, "bert_score_f1": 0.0}
            count = 0
            
            for pred, ref in zip(predictions, references):
                try:
                    P, R, F1 = score([pred], [ref], 
                                   model_type=self.model_type,
                                   num_layers=self.num_layers,
                                   lang='en', 
                                   verbose=False)
                    total_scores["bert_score_precision"] += P.item()
                    total_scores["bert_score_recall"] += R.item()
                    total_scores["bert_score_f1"] += F1.item()
                    count += 1
                except Exception as e:
                    print(f"[WARNING] Individual BERTScore evaluation failed: {e}")
            
            if count > 0:
                avg_precision = total_scores["bert_score_precision"] / count
                avg_recall = total_scores["bert_score_recall"] / count
                avg_f1 = total_scores["bert_score_f1"] / count
            else:
                avg_precision = avg_recall = avg_f1 = 0.0
        
        # Initialize results with batch scores
        results = {
            "bert_score_precision": avg_precision,
            "bert_score_recall": avg_recall,
            "bert_score_f1": avg_f1
        }
        
        # Context-based evaluation (if contexts provided)
        if contexts:
            context_scores = {"bert_answer_context_f1": 0.0, "bert_reference_context_f1": 0.0, "bert_context_consistency": 0.0}
            count = len(predictions)
            
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                context = contexts[i] if i < len(contexts) else None
                if context:
                    ctx_scores = self._evaluate_context_bert_scores(pred, ref, context)
                    for key, value in ctx_scores.items():
                        context_scores[key] += value
            
            # Average context scores
            for key in context_scores:
                context_scores[key] = context_scores[key] / count if count > 0 else 0.0
            
            results.update(context_scores)
        else:
            results.update({
                "bert_answer_context_f1": 0.0,
                "bert_reference_context_f1": 0.0,
                "bert_context_consistency": 0.0
            })
        
        # Calculate composite score
        results["composite_bert_score"] = self._calculate_composite_score(results)
        
        return results

def find_json_files(directory: str = "dataqa") -> List[str]:
    """Find JSON files in the dataqa directory"""
    json_files = []
    
    # Check dataqa directory only
    if os.path.exists(directory):
        json_files = glob.glob(os.path.join(directory, "*.json"))
    
    return sorted(json_files)

def select_json_file(json_files: List[str]) -> Optional[str]:
    """Allow user to select which JSON file to use"""
    if not json_files:
        print("[ERROR] No JSON files found in dataqa directory")
        return None
    
    if len(json_files) == 1:
        print(f"[INFO] Found JSON file: {json_files[0]}")
        return json_files[0]
    
    print("\n📁 Available JSON files:")
    for i, file_path in enumerate(json_files, 1):
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        print(f"  {i}. {file_path} ({file_size:,} bytes)")
    
    while True:
        try:
            choice = input(f"\nSelect file (1-{len(json_files)}) or press Enter for first file: ").strip()
            
            if not choice:
                return json_files[0]
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(json_files):
                return json_files[choice_idx]
            else:
                print(f"Please enter a number between 1 and {len(json_files)}")
        
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n[INFO] Evaluation cancelled")
            return None

def main():
    """Main function for flexible BERTScore evaluation"""
    print("=" * 80)
    print("DATAQA JSON BERTSCORE EVALUATION")
    print("=" * 80)
    
    # Find available JSON files in dataqa directory
    json_files = find_json_files()
    
    # Let user select which file to evaluate
    eval_file = select_json_file(json_files)
    if not eval_file:
        return
    
    # Initialize evaluator
    evaluator = BERTScoreEvaluator(verbose=True)
    
    # Load evaluation data
    print(f"[INFO] Loading evaluation data from {eval_file}...")
    try:
        with open(eval_file, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
        print(f"[INFO] Successfully loaded {len(eval_data)} evaluation items")
    except Exception as e:
        print(f"[ERROR] Failed to load evaluation data: {e}")
        return
    
    if not eval_data:
        print("[ERROR] No evaluation data found")
        return
    
    # Extract predictions, references, and contexts
    predictions = []
    references = []
    contexts = []
    
    for item in eval_data:
        if "predicted_answer" in item and "reference_answer" in item:
            predictions.append(item["predicted_answer"])
            references.append(item["reference_answer"])
            # Add context if available
            if "context" in item and item["context"]:
                contexts.append(item["context"] if isinstance(item["context"], list) else [item["context"]])
            else:
                contexts.append([])
    
    if not predictions or not references:
        print("[ERROR] No valid prediction-reference pairs found")
        return
    
    print(f"Evaluating {len(predictions)} answer pairs using BERTScore...")
    print("-" * 80)
    
    # Evaluate batch
    avg_scores = evaluator.evaluate_batch(predictions, references, contexts if any(contexts) else None)
    
    print("BERTScore Evaluation Results:")
    print(f"  BERTScore Precision:            {avg_scores['bert_score_precision']:.4f}")
    print(f"  BERTScore Recall:               {avg_scores['bert_score_recall']:.4f}")
    print(f"  BERTScore F1:                   {avg_scores['bert_score_f1']:.4f}")
    print(f"  BERT Answer-Context F1:         {avg_scores['bert_answer_context_f1']:.4f}")
    print(f"  BERT Reference-Context F1:      {avg_scores['bert_reference_context_f1']:.4f}")
    print(f"  BERT Context Consistency:       {avg_scores['bert_context_consistency']:.4f}")
    print(f"  Composite BERT Score:           {avg_scores['composite_bert_score']:.4f}")
    
    # Generate output filename based on input file
    base_name = os.path.splitext(os.path.basename(eval_file))[0]
    results_file = f"{base_name}_bert_score_results.json"
    
    # Save detailed results to file
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": avg_scores,
                "total_evaluations": len(predictions),
                "evaluation_timestamp": str(datetime.datetime.now()),
                "source_file": eval_file,
                "model_type": evaluator.model_type or "default",
                "num_layers": evaluator.num_layers or "default",
                "evaluation_type": "bert_score"
            }, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Detailed results saved to: {results_file}")
    except Exception as e:
        print(f"[WARNING] Failed to save results: {e}")
    
    print(f"\n📊 Evaluation Summary:")
    print(f"  Source: {eval_file}")
    print(f"  Items evaluated: {len(predictions)}")
    print(f"  Output: {results_file}")
    
    print("\n" + "=" * 80)
    print("DATAQA JSON BERTSCORE EVALUATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
