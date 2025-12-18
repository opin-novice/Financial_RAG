#!/usr/bin/env python3
"""
Inference Time Measurement Script
==================================
Measures only the inference time (time to produce answers) for 5 questions
from Policy_Related.json using main2.py
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime

# Import RAG system from main2
from main2 import CoreRAGSystem

def load_questions(json_path: str, num_questions: int = 5):
    """Load questions from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Return first num_questions
    return data[:num_questions]

def measure_inference_time(rag_system, question: str):
    """Measure inference time for a single question"""
    start_time = time.time()
    result = rag_system.process_query(question)
    end_time = time.time()
    
    inference_time = end_time - start_time
    return inference_time, result

def main():
    """Main function to measure inference times"""
    print("="*70)
    print("Inference Time Measurement Script")
    print("="*70)
    
    # Path to Policy_Related.json
    json_path = Path("domain_based_qa/Policy_Related.json")
    
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    
    # Load 5 questions
    print(f"\nLoading questions from: {json_path}")
    questions_data = load_questions(str(json_path), num_questions=5)
    print(f"Loaded {len(questions_data)} questions\n")
    
    # Initialize RAG system
    print("Initializing RAG system...")
    try:
        rag_system = CoreRAGSystem()
        print("RAG system initialized successfully\n")
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        sys.exit(1)
    
    # Measure inference time for each question
    print("="*70)
    print("Measuring Inference Times")
    print("="*70)
    
    results = []
    total_time = 0.0
    
    for i, qdata in enumerate(questions_data, 1):
        question = qdata['question']
        print(f"\nQuestion {i}/5:")
        print(f"   {question[:80]}{'...' if len(question) > 80 else ''}")
        
        # Measure inference time
        inference_time, result = measure_inference_time(rag_system, question)
        total_time += inference_time
        
        results.append({
            'question_id': i,
            'question': question,
            'inference_time_seconds': inference_time,
            'response_length': len(result.get('response', ''))
        })
        
        print(f"   Inference Time: {inference_time:.2f} seconds")
        print(f"   Response Length: {len(result.get('response', ''))} characters")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nTotal Questions: {len(results)}")
    print(f"Total Inference Time: {total_time:.2f} seconds")
    print(f"Average Inference Time: {total_time/len(results):.2f} seconds")
    print(f"Min Inference Time: {min(r['inference_time_seconds'] for r in results):.2f} seconds")
    print(f"Max Inference Time: {max(r['inference_time_seconds'] for r in results):.2f} seconds")
    
    print("\n" + "="*70)
    print("Detailed Results")
    print("="*70)
    
    for r in results:
        print(f"\nQuestion {r['question_id']}:")
        print(f"  Question: {r['question'][:60]}...")
        print(f"  Inference Time: {r['inference_time_seconds']:.2f} seconds")
        print(f"  Response Length: {r['response_length']} characters")
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"inference_time_results_{timestamp}.json"
    
    output_data = {
        'test_file': str(json_path),
        'num_questions': len(results),
        'total_inference_time_seconds': total_time,
        'average_inference_time_seconds': total_time/len(results),
        'min_inference_time_seconds': min(r['inference_time_seconds'] for r in results),
        'max_inference_time_seconds': max(r['inference_time_seconds'] for r in results),
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print("\nMeasurement completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
