#!/usr/bin/env python3
"""
Domain-Based Vanilla RAG Cosine Similarity Evaluation
=====================================================
Evaluates Vanilla RAG system on specific domain QA pairs using cosine similarity
Usage: python domain_vanilla_rag_cosine_evaluation.py --domain_file "factual_questions.json"
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

def load_domain_qa(domain_file):
    """Load domain-specific QA data"""
    domain_path = os.path.join('domain_based_qa', domain_file)
    if not os.path.exists(domain_path):
        raise FileNotFoundError(f"Domain file not found: {domain_path}")
    
    with open(domain_path, 'r', encoding='utf-8') as f:
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

def get_vanilla_rag_responses(questions, domain_name):
    """Get responses from Vanilla RAG system"""
    try:
        from main import VanillaRAGSystem
        print("✅ Vanilla RAG system imported successfully")
        
        rag_system = VanillaRAGSystem()
        print("✅ Vanilla RAG system initialized successfully")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Using device: {device}")
        
        responses = []
        total_questions = len(questions)
        
        print(f"�� Processing {total_questions} questions from {domain_name} domain...")
        
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
                    'question': question_text,
                    'answer': qa_pair['answer'],
                    'raw_response': f"Error: {e}",
                    'response': f"Error: {e}",
                    'processing_time': 0,
                    'device_used': device
                })
        
        print(f"✅ Successfully processed {len(responses)} questions")
        return responses
        
    except ImportError as e:
        print(f"❌ Failed to import Vanilla RAG system: {e}")
        return []
    except Exception as e:
        print(f"❌ Failed to initialize Vanilla RAG system: {e}")
        return []

def calculate_cosine_similarity(responses, device="cuda"):
    """Calculate cosine similarity between responses and ground truth answers"""
    print(f"�� Calculating Cosine Similarity on {device.upper()}...")
    
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

def print_summary_stats(similarity_results, total_questions):
    """Print summary statistics in preferred format"""
    print("\n" + "="*60)
    print("📊 DOMAIN EVALUATION SUMMARY - VANILLA RAG (COSINE SIMILARITY)")
    print("="*60)
    
    print(f"Mean Similarity: {similarity_results['mean_similarity']:.4f}")
    print(f"Median Similarity: {similarity_results['median_similarity']:.4f}")
    print(f"Std Deviation: {similarity_results['std_similarity']:.4f}")
    print(f"Min Similarity: {similarity_results['min_similarity']:.4f}")
    print(f"Max Similarity: {similarity_results['max_similarity']:.4f}")
    print(f"Total Questions: {similarity_results['total_questions']}")

def main():
    """Main function for domain-based cosine similarity evaluation"""
    parser = argparse.ArgumentParser(description='Domain-Based Vanilla RAG Cosine Similarity Evaluation')
    parser.add_argument('--domain_file', required=True, help='Domain JSON file name (e.g., "factual_questions.json")')
    args = parser.parse_args()
    
    domain_name = args.domain_file.replace('.json', '')
    
    print(f"�� Starting Vanilla RAG Domain-Based Cosine Similarity Evaluation")
    print(f"📁 Domain: {domain_name}")
    print("=" * 70)
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    device = "cuda" if gpu_available else "cpu"
    
    # Load domain data
    print(f"\n📂 Loading domain QA pairs from {args.domain_file}...")
    try:
        questions = load_domain_qa(args.domain_file)
        print(f"✅ Loaded {len(questions)} questions from {domain_name} domain")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Get Vanilla RAG responses
    print(f"\n🤖 Getting Vanilla RAG responses...")
    responses = get_vanilla_rag_responses(questions, domain_name)
    
    if not responses:
        print("❌ No responses generated. Exiting.")
        return
    
    # Calculate cosine similarity
    print(f"\n�� Calculating cosine similarity...")
    similarity_results = calculate_cosine_similarity(responses, device)
    
    if similarity_results is None:
        print("❌ Cosine similarity calculation failed. Exiting.")
        return
    
    # Print summary statistics
    print_summary_stats(similarity_results, len(questions))
    
    # Prepare results
    results = {
        'evaluation_info': {
            'timestamp': datetime.now().isoformat(),
            'model': 'Vanilla RAG (VanillaRAGSystem)',
            'domain': domain_name,
            'domain_file': args.domain_file,
            'total_questions': len(questions),
            'successful_responses': len(similarity_results['valid_responses']),
            'device_used': device,
            'gpu_available': gpu_available,
            'evaluation_type': 'cosine_similarity'
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
            'processing_time': resp['processing_time']
        })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"vanilla_rag_domain_cosine_{domain_name}_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {output_file}")

if __name__ == "__main__":
    main()