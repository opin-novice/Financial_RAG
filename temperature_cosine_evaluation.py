#!/usr/bin/env python3
"""
Advanced RAG Temperature-based Cosine Similarity Evaluation Script
================================================================
Evaluates Advanced RAG system across different temperature values using cosine similarity
Usage: python advanced_rag_temperature_evaluation.py --temperature 0.1
"""

import os
import json
import time
import torch
import argparse
import statistics
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List

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

def load_qa_pairs(file_path: str) -> List[Dict]:
    """Load QA pairs from JSON file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"QA pairs file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
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

def get_advanced_rag_responses(questions: List[Dict], temperature: float) -> List[Dict]:
    """Get responses from Advanced RAG system with specified temperature"""
    try:
        from main2 import CoreRAGSystem as AdvancedRAGSystem
        print("✅ Advanced RAG system imported successfully")
        
        # Temporarily modify the temperature in the config
        import main2
        original_temp = main2.OLLAMA_TEMPERATURE
        main2.OLLAMA_TEMPERATURE = temperature
        
        # Initialize Advanced RAG
        rag_system = AdvancedRAGSystem()
        print(f"✅ Advanced RAG system initialized with temperature: {temperature}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Using device: {device}")
        
        responses = []
        total_questions = len(questions)
        
        print(f" Processing {total_questions} questions with temperature {temperature}...")
        
        for i, qa_pair in enumerate(questions):
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
                    'answer': qa_pair['answer'],
                    'response': response_text,
                    'processing_time': processing_time,
                    'temperature': temperature
                })
                
                print(f"   ✅ Response generated in {processing_time:.2f}s")
                
            except Exception as e:
                print(f"   ❌ Error processing question {i+1}: {str(e)}")
                responses.append({
                    'question': question_text,
                    'answer': qa_pair['answer'],
                    'response': f"Error: {str(e)}",
                    'processing_time': 0,
                    'temperature': temperature
                })
        
        # Restore original temperature
        main2.OLLAMA_TEMPERATURE = original_temp
        
        print(f"✅ Successfully processed {len(responses)} questions")
        return responses
        
    except Exception as e:
        print(f"❌ Error initializing Advanced RAG system: {str(e)}")
        return []

def calculate_cosine_similarity(responses: List[Dict], device: str = "cuda") -> Dict:
    """Calculate cosine similarity between responses and ground truth answers"""
    print(f" Calculating Cosine Similarity on {device.upper()}...")
    
    try:
        # Initialize embedding model
        model_name = "BAAI/bge-m3"  # Same as your RAG system
        embedding_model = SentenceTransformer(model_name, device=device)
        
        similarities = []
        valid_responses = []
        
        for i, resp in enumerate(responses):
            if resp['response'].startswith("Error:"):
                print(f"   Skipping question {i+1} due to error")
                continue
                
            try:
                # Get embeddings
                response_embeddings = embedding_model.encode([resp['response']], normalize_embeddings=True)
                ground_truth_embeddings = embedding_model.encode([resp['answer']], normalize_embeddings=True)
                
                # Calculate cosine similarity
                similarity = cosine_similarity(response_embeddings, ground_truth_embeddings)[0][0]
                similarities.append(float(similarity))
                valid_responses.append(resp)
                
                print(f"   Question {i+1}: Similarity = {similarity:.4f}")
                
            except Exception as e:
                print(f"   Error calculating similarity for question {i+1}: {e}")
                continue
        
        if not similarities:
            print("❌ No valid similarities calculated")
            return None
        
        # Calculate statistics
        mean_similarity = statistics.mean(similarities)
        median_similarity = statistics.median(similarities)
        std_similarity = statistics.stdev(similarities) if len(similarities) > 1 else 0
        min_similarity = min(similarities)
        max_similarity = max(similarities)
        
        print(f"✅ Cosine similarity calculation completed")
        print(f"   Mean Similarity: {mean_similarity:.4f}")
        print(f"   Median Similarity: {median_similarity:.4f}")
        print(f"   Std Deviation: {std_similarity:.4f}")
        print(f"   Min Similarity: {min_similarity:.4f}")
        print(f"   Max Similarity: {max_similarity:.4f}")
        
        return {
            'similarities': similarities,
            'mean_similarity': mean_similarity,
            'median_similarity': median_similarity,
            'std_similarity': std_similarity,
            'min_similarity': min_similarity,
            'max_similarity': max_similarity,
            'total_questions': len(similarities),
            'valid_responses': valid_responses
        }
        
    except Exception as e:
        print(f"❌ Error calculating cosine similarity: {str(e)}")
        return None

def print_summary_stats(similarity_results: Dict, temperature: float):
    """Print summary statistics in preferred format"""
    print("\n" + "="*70)
    print(f"📊 ADVANCED RAG TEMPERATURE EVALUATION SUMMARY (T={temperature})")
    print("="*70)
    
    print(f"Mean Similarity: {similarity_results['mean_similarity']:.4f}")
    print(f"Median Similarity: {similarity_results['median_similarity']:.4f}")
    print(f"Std Deviation: {similarity_results['std_similarity']:.4f}")
    print(f"Min Similarity: {similarity_results['min_similarity']:.4f}")
    print(f"Max Similarity: {similarity_results['max_similarity']:.4f}")
    print(f"Total Questions: {similarity_results['total_questions']}")

def evaluate_temperature(temperature: float, qa_pairs: List[Dict], device: str) -> Dict:
    """Evaluate Advanced RAG with specified temperature"""
    print(f"\n🚀 Starting Advanced RAG evaluation with temperature {temperature}")
    print("="*70)
    
    # Get responses from Advanced RAG
    responses = get_advanced_rag_responses(qa_pairs, temperature)
    
    if not responses:
        print("❌ No responses generated. Exiting.")
        return None
    
    # Calculate cosine similarity
    print(f"\n Calculating cosine similarity...")
    similarity_results = calculate_cosine_similarity(responses, device)
    
    if similarity_results is None:
        print("❌ Cosine similarity calculation failed. Exiting.")
        return None
    
    # Print summary statistics
    print_summary_stats(similarity_results, temperature)
    
    # Prepare results
    results = {
        'evaluation_info': {
            'timestamp': datetime.now().isoformat(),
            'model': 'Advanced RAG (CoreRAGSystem)',
            'temperature': temperature,
            'total_questions': len(qa_pairs),
            'successful_responses': len(similarity_results['valid_responses']),
            'device_used': device,
            'gpu_available': torch.cuda.is_available(),
            'evaluation_type': 'cosine_similarity_temperature'
        },
        'cosine_similarity_metrics': {
            'mean_similarity': similarity_results['mean_similarity'],
            'median_similarity': similarity_results['median_similarity'],
            'std_similarity': similarity_results['std_similarity'],
            'min_similarity': similarity_results['min_similarity'],
            'max_similarity': similarity_results['max_similarity'],
            'total_questions': similarity_results['total_questions'],
            'similarities': similarity_results['similarities']
        },
        'detailed_results': []
    }
    
    # Add detailed results
    for i, resp in enumerate(similarity_results['valid_responses']):
        results['detailed_results'].append({
            'question_id': i,
            'question': resp['question'],
            'ground_truth': resp['answer'],
            'prediction': resp['response'],
            'cosine_similarity': similarity_results['similarities'][i],
            'processing_time': resp['processing_time'],
            'temperature': temperature
        })
    
    return results

def main():
    """Main function for Advanced RAG temperature-based cosine similarity evaluation"""
    parser = argparse.ArgumentParser(description='Advanced RAG Temperature-based Cosine Similarity Evaluation')
    parser.add_argument('--temperature', type=float, required=True, 
                       help='Temperature value to test (e.g., 0.1, 0.4, 0.7, 1.0)')
    parser.add_argument('--qa_file', default='totalqapair.json', 
                       help='QA pairs JSON file (default: totalqapair.json)')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Number of questions to sample (default: all questions)')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Advanced RAG Temperature-based Cosine Similarity Evaluation")
    print(f"🌡️  Temperature: {args.temperature}")
    print(f"📄 QA File: {args.qa_file}")
    print("=" * 70)
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    device = "cuda" if gpu_available else "cpu"
    
    # Load QA pairs
    print(f"\n📂 Loading QA pairs from {args.qa_file}...")
    try:
        qa_pairs = load_qa_pairs(args.qa_file)
        print(f"✅ Loaded {len(qa_pairs)} QA pairs")
        
        # Sample if requested
        if args.sample_size and args.sample_size < len(qa_pairs):
            import random
            qa_pairs = random.sample(qa_pairs, args.sample_size)
            print(f"📊 Sampled {len(qa_pairs)} questions for evaluation")
            
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Run evaluation
    results = evaluate_temperature(args.temperature, qa_pairs, device)
    
    if results is None:
        print("❌ Evaluation failed. Exiting.")
        return
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"advanced_rag_temperature_{args.temperature}_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Print final summary
    print("\n" + "="*70)
    print("🎯 FINAL SUMMARY")
    print("="*70)
    print(f"System: Advanced RAG")
    print(f"Temperature: {args.temperature}")
    print(f"Mean Similarity: {results['cosine_similarity_metrics']['mean_similarity']:.4f}")
    print(f"Median Similarity: {results['cosine_similarity_metrics']['median_similarity']:.4f}")
    print(f"Std Deviation: {results['cosine_similarity_metrics']['std_similarity']:.4f}")
    print(f"Min Similarity: {results['cosine_similarity_metrics']['min_similarity']:.4f}")
    print(f"Max Similarity: {results['cosine_similarity_metrics']['max_similarity']:.4f}")
    print(f"Total Questions: {results['cosine_similarity_metrics']['total_questions']}")

if __name__ == "__main__":
    main()