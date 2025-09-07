#!/usr/bin/env python3
"""
GPU-Accelerated Advanced RAG BERTscore Evaluation for BNqapairs (10 questions)
================================================================================
Uses GPU acceleration and processes only 10 questions for speed
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

def get_advanced_rag_responses_gpu(questions, max_questions=10):
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
        total_questions = min(max_questions, len(questions))
        
        print(f" Processing {total_questions} questions on {device.upper()}...")
        
        for i, question in enumerate(questions[:total_questions]):
            print(f"Processing question {i+1}/{total_questions}: {question[:50]}...")
            
            try:
                start_time = time.time()
                raw_response = rag_system.process_query(question)
                
                # Extract text from response
                response_text = extract_text_from_response(raw_response)
                
                processing_time = time.time() - start_time
                
                responses.append({
                    'question': question,
                    'raw_response': raw_response,
                    'response': response_text,
                    'processing_time': processing_time,
                    'device_used': device
                })
                
                print(f"✅ Question {i+1} completed in {processing_time:.2f}s on {device.upper()}")
                print(f"   Response preview: {response_text[:100]}...")
                
            except Exception as e:
                print(f"❌ Error processing question {i+1}: {e}")
                responses.append({
                    'question': question,
                    'raw_response': f"Error: {str(e)}",
                    'response': f"Error: {str(e)}",
                    'processing_time': 0,
                    'device_used': device
                })
        
        return responses
        
    except ImportError as e:
        print(f"❌ Failed to import Advanced RAG system: {e}")
        return []
    except Exception as e:
        print(f"❌ Failed to initialize Advanced RAG system: {e}")
        return []

def run_bertscore_evaluation_gpu(responses, ground_truth_answers):
    """Run BERTscore evaluation with GPU optimization"""
    print("\n🔄 Running BERTscore evaluation with GPU acceleration...")
    
    # Extract predictions and references
    predictions = [resp['response'] for resp in responses]
    references = ground_truth_answers[:len(predictions)]
    
    # Ensure all predictions are strings
    predictions = [str(pred) for pred in predictions]
    references = [str(ref) for ref in references]
    
    print(f"Evaluating {len(predictions)} prediction-reference pairs...")
    print(f"Sample prediction: {predictions[0][:100]}...")
    print(f"Sample reference: {references[0][:100]}...")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  BERTscore using device: {device}")
        
        # Run BERTscore with GPU acceleration
        P, R, F1 = score(
            predictions, 
            references, 
            lang="bn", 
            verbose=True,
            device=device
        )
        
        # Calculate statistics
        precision = P.mean().item()
        recall = R.mean().item()
        f1_score = F1.mean().item()
        
        precision_std = P.std().item()
        recall_std = R.std().item()
        f1_std = F1.std().item()
        
        results = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'precision_std': precision_std,
            'recall_std': recall_std,
            'f1_std': f1_std,
            'num_questions': len(predictions),
            'device_used': device,
            'detailed_scores': {
                'precision_scores': P.tolist(),
                'recall_scores': R.tolist(),
                'f1_scores': F1.tolist()
            }
        }
        
        print(f"✅ BERTscore evaluation completed on {device.upper()}!")
        print(f"Precision: {precision:.4f} ± {precision_std:.4f}")
        print(f"Recall: {recall:.4f} ± {recall_std:.4f}")
        print(f"F1-Score: {f1_score:.4f} ± {f1_std:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ BERTscore evaluation failed: {e}")
        return None

def main():
    """Main function with GPU optimization for 10 questions"""
    print(" Starting GPU-Accelerated Advanced RAG BERTscore Evaluation for BNqapairs (10 questions)")
    print("=" * 80)
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    # Load data
    print("\n📄 Loading BNqapairs data...")
    bnqapairs = load_bnqapairs()
    print(f"✅ Loaded {len(bnqapairs)} question-answer pairs")
    
    # Extract questions and answers
    questions = [item['question'] for item in bnqapairs]
    ground_truth_answers = [item['answer'] for item in bnqapairs]
    
    # Get Advanced RAG responses (10 questions only)
    print(f"\n�� Getting Advanced RAG responses for 10 questions...")
    responses = get_advanced_rag_responses_gpu(questions, max_questions=10)
    
    if not responses:
        print("❌ No responses generated. Exiting.")
        return
    
    # Run BERTscore evaluation
    bertscore_results = run_bertscore_evaluation_gpu(responses, ground_truth_answers)
    
    if bertscore_results is None:
        print("❌ BERTscore evaluation failed. Exiting.")
        return
    
    # Prepare final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'evaluation_info': {
            'timestamp': timestamp,
            'dataset': 'BNqapairs (10 questions)',
            'system': 'Advanced RAG (GPU-Accelerated)',
            'total_questions': len(questions),
            'evaluated_questions': len(responses),
            'language': 'Bengali (bn)',
            'gpu_available': gpu_available,
            'device_used': bertscore_results.get('device_used', 'unknown')
        },
        'bertscore_results': bertscore_results,
        'detailed_responses': responses
    }
    
    # Save results
    output_file = f"advanced_rag_bertscore_bnqapairs_gpu_10q_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    print("\n📊 Summary:")
    print(f"Dataset: BNqapairs (10 questions)")
    print(f"System: Advanced RAG (GPU-Accelerated)")
    print(f"Device: {bertscore_results.get('device_used', 'unknown').upper()}")
    print(f"F1-Score: {bertscore_results['f1_score']:.4f} ± {bertscore_results['f1_std']:.4f}")
    print(f"Precision: {bertscore_results['precision']:.4f} ± {bertscore_results['precision_std']:.4f}")
    print(f"Recall: {bertscore_results['recall']:.4f} ± {bertscore_results['recall_std']:.4f}")

if __name__ == "__main__":
    main()