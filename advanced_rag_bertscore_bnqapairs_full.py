#!/usr/bin/env python3
"""
GPU-Accelerated Advanced RAG BERTscore Evaluation for BNqapairs (Full Dataset)
================================================================================
Uses GPU acceleration and processes all 48 Bengali questions
"""

import os
import json
import time
import torch
from datetime import datetime
from bert_score import score

def check_gpu_availability():
    """Check if GPU is available and show info"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU Available: {gpu_name}")
        print(f"   GPU Count: {gpu_count}")
        print(f"   GPU Memory: {gpu_memory:.1f} GB")
        return True
    else:
        print("❌ GPU not available, falling back to CPU")
        return False

def load_bnqapairs():
    """Load BNqapairs data"""
    with open('dataqa/BNqapair.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_text_from_response(response):
    """Extract text from response (handles both string and dict)"""
    if isinstance(response, str):
        return response
    elif isinstance(response, dict):
        # Try common keys for text content
        for key in ['answer', 'response', 'text', 'content', 'result']:
            if key in response and isinstance(response[key], str):
                return response[key]
        # If no text found, convert dict to string
        return str(response)
    else:
        return str(response)

def get_advanced_rag_responses_gpu(questions, max_questions=None):
    """Get responses from Advanced RAG system with GPU optimization"""
    try:
        from main2 import CoreRAGSystem as AdvancedRAGSystem
        print("✅ Advanced RAG system imported successfully")
        
        # Initialize Advanced RAG
        rag_system = AdvancedRAGSystem()
        print("✅ Advanced RAG system initialized successfully")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Using device: {device}")
        
        responses = []
        total_questions = len(questions) if max_questions is None else min(max_questions, len(questions))
        
        print(f" Processing {total_questions} questions on {device.upper()}...")
        
        for i, qa_pair in enumerate(questions[:total_questions]):
            # Extract question text from the dictionary
            question_text = qa_pair['question']
            print(f"Processing question {i+1}/{total_questions}: {question_text[:50]}...")
            
            try:
                start_time = time.time()
                raw_response = rag_system.process_query(question_text)
                
                # Extract text from response
                response_text = extract_text_from_response(raw_response)
                
                processing_time = time.time() - start_time
                
                responses.append({
                    'question': question_text,
                    'answer': qa_pair['answer'],  # Store the ground truth answer
                    'response': response_text,
                    'processing_time': processing_time
                })
                
                print(f"   ✅ Response generated in {processing_time:.2f}s")
                print(f"   Response preview: {response_text[:100]}...")
                
            except Exception as e:
                print(f"   ❌ Error processing question {i+1}: {str(e)}")
                responses.append({
                    'question': question_text,
                    'answer': qa_pair['answer'],
                    'response': f"Error: {str(e)}",
                    'processing_time': 0
                })
        
        print(f"✅ Successfully processed {len(responses)} questions")
        return responses
        
    except Exception as e:
        print(f"❌ Error initializing Advanced RAG system: {str(e)}")
        return []

def calculate_bertscore(predictions, references, device="cuda"):
    """Calculate BERTScore with GPU acceleration"""
    print(f"🔍 Calculating BERTScore on {device.upper()}...")
    
    try:
        # Calculate BERTScore
        P, R, F1 = score(predictions, references, 
                        model_type="bert-base-uncased",
                        device=device,
                        verbose=True)
        
        # Convert to lists for easier handling
        precision_scores = P.tolist()
        recall_scores = R.tolist()
        f1_scores = F1.tolist()
        
        # Calculate statistics - FIX: Use original tensors for mean calculation
        avg_precision = P.mean().item()
        avg_recall = R.mean().item()
        avg_f1 = F1.mean().item()
        
        print(f"✅ BERTScore calculation completed")
        print(f"   Average Precision: {avg_precision:.4f}")
        print(f"   Average Recall: {avg_recall:.4f}")
        print(f"   Average F1: {avg_f1:.4f}")
        
        return {
            'precision_scores': precision_scores,
            'recall_scores': recall_scores,
            'f1_scores': f1_scores,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1
        }
        
    except Exception as e:
        print(f"❌ Error calculating BERTScore: {str(e)}")
        return None

def main():
    """Main function with GPU optimization for full dataset"""
    print("🚀 Starting GPU-Accelerated Advanced RAG BERTscore Evaluation for BNqapairs (Full Dataset)")
    print("=" * 80)
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    device = "cuda" if gpu_available else "cpu"
    
    # Load data
    print("\n Loading BNqapairs dataset...")
    questions = load_bnqapairs()
    print(f"✅ Loaded {len(questions)} questions")
    
    # Get Advanced RAG responses (all questions)
    print(f"\n Getting Advanced RAG responses for all {len(questions)} questions...")
    responses = get_advanced_rag_responses_gpu(questions, max_questions=None)
    
    if not responses:
        print("❌ No responses generated. Exiting.")
        return
    
    # Prepare data for BERTScore
    print("\n Preparing data for BERTScore evaluation...")
    predictions = [r['response'] for r in responses]
    references = [r['answer'] for r in responses]  # Use stored answers from responses
    
    print(f"📝 Predictions: {len(predictions)}")
    print(f"📝 References: {len(references)}")
    print(f"Sample prediction: {predictions[0][:100]}...")
    print(f"Sample reference: {references[0][:100]}...")
    
    # Calculate BERTScore
    bertscore_results = calculate_bertscore(predictions, references, device)
    
    if bertscore_results is None:
        print("❌ BERTScore calculation failed. Exiting.")
        return
    
    # Prepare results
    results = {
        'evaluation_info': {
            'timestamp': datetime.now().isoformat(),
            'model': 'Advanced RAG (CoreRAGSystem)',
            'dataset': 'BNqapairs (Full Dataset)',
            'total_questions': len(questions),
            'successful_responses': len(responses),
            'device_used': device,
            'gpu_available': gpu_available
        },
        'bertscore_metrics': {
            'avg_precision': bertscore_results['avg_precision'],
            'avg_recall': bertscore_results['avg_recall'],
            'avg_f1': bertscore_results['avg_f1'],
            'precision_scores': bertscore_results['precision_scores'],
            'recall_scores': bertscore_results['recall_scores'],
            'f1_scores': bertscore_results['f1_scores']
        },
        'detailed_results': []
    }
    
    # Add detailed results
    for i, resp in enumerate(responses):
        results['detailed_results'].append({
            'question_id': i,
            'question': resp['question'],
            'ground_truth': resp['answer'],
            'prediction': resp['response'],
            'precision': bertscore_results['precision_scores'][i],
            'recall': bertscore_results['recall_scores'][i],
            'f1': bertscore_results['f1_scores'][i],
            'processing_time': resp['processing_time']
        })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"advanced_rag_bertscore_bnqapairs_full_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {output_file}")
    print(f"Dataset: BNqapairs (Full Dataset)")
    print(f"Total Questions: {len(questions)}")
    print(f"Successful Responses: {len(responses)}")
    print(f"Average Precision: {bertscore_results['avg_precision']:.4f}")
    print(f"Average Recall: {bertscore_results['avg_recall']:.4f}")
    print(f"Average F1: {bertscore_results['avg_f1']:.4f}")

if __name__ == "__main__":
    main()