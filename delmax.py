#!/usr/bin/env python3
"""
Calculate Mean of Max Similarity from RAG Cosine Similarity Evaluation Results
"""

import json
import numpy as np
import argparse
from typing import List, Dict, Any

def load_evaluation_results(file_path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in file '{file_path}': {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading file '{file_path}': {e}")
        return None

def extract_similarities(data: Dict[str, Any], metric_type: str = 'response') -> List[float]:
    """Extract similarity values from evaluation results"""
    similarities = []
    
    if 'results' not in data:
        print("❌ Error: No 'results' key found in JSON data")
        return []
    
    results = data['results']
    print(f"📊 Found {len(results)} evaluation results")
    
    for i, result in enumerate(results):
        if metric_type == 'response':
            key = 'response_similarity'
        elif metric_type == 'passages':
            key = 'passages_similarity'
        else:
            key = 'max_similarity'
            
        if key in result:
            sim_value = result[key]
            if isinstance(sim_value, (int, float)):
                similarities.append(float(sim_value))
            else:
                print(f"⚠️ Warning: Invalid {key} value at index {i}: {sim_value}")
        else:
            print(f"⚠️ Warning: No {key} found at index {i}")
    
    return similarities

def calculate_statistics(max_similarities: List[float]) -> Dict[str, float]:
    """Calculate statistics for max_similarity values"""
    if not max_similarities:
        return {}
    
    stats = {
        'mean': np.mean(max_similarities),
        'median': np.median(max_similarities),
        'std': np.std(max_similarities),
        'min': np.min(max_similarities),
        'max': np.max(max_similarities),
        'count': len(max_similarities)
    }
    
    return stats

def print_results(stats: Dict[str, float], file_path: str, metric_name: str = "RESPONSE SIMILARITY"):
    """Print formatted results"""
    print("\n" + "="*60)
    print(f"📈 {metric_name} STATISTICS")
    print(f"📄 File: {file_path}")
    print("="*60)
    
    if not stats:
        print("❌ No valid data to analyze")
        return
    
    print(f"📊 Total Results: {stats['count']}")
    print(f"📊 Valid Similarities: {stats['count']}")
    print()
    print(f"📈 Mean:     {stats['mean']:.4f}")
    print(f"📈 Median:   {stats['median']:.4f}")
    print(f"📈 Std Dev:  {stats['std']:.4f}")
    print(f"📈 Min:      {stats['min']:.4f}")
    print(f"📈 Max:      {stats['max']:.4f}")
    print()
    
    # Quality assessment
    mean_score = stats['mean']
    if mean_score >= 0.8:
        quality = "🟢 Excellent"
    elif mean_score >= 0.6:
        quality = "🟡 Good"
    elif mean_score >= 0.4:
        quality = "🟠 Fair"
    elif mean_score >= 0.2:
        quality = "🔴 Poor"
    else:
        quality = "🔴 Very Poor"
    
    print(f"🎯 Overall Quality: {quality} (Mean: {mean_score:.4f})")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="Calculate mean of max_similarity from RAG cosine similarity evaluation results"
    )
    parser.add_argument(
        'file', 
        nargs='?', 
        default='logs//rag_cosine_similarity_20250831_221104.json',
        help='Path to the evaluation results JSON file'
    )
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true',
        help='Show verbose output including individual values'
    )
    
    args = parser.parse_args()
    
    print("🚀 Loading evaluation results...")
    
    # Load data
    data = load_evaluation_results(args.file)
    if data is None:
        return 1
    
    # Extract similarities based on metric type
    similarities = extract_similarities(data, 'response')
    if not similarities:
        print("❌ No valid response_similarity values found")
        return 1
    
    # Calculate statistics
    stats = calculate_statistics(similarities)
    
    # Print results
    print_results(stats, args.file, "RESPONSE SIMILARITY")
    
    # Verbose output
    if args.verbose:
        print("\n📋 Individual Response Similarity Values:")
        print("-" * 40)
        for i, value in enumerate(similarities[:20]):  # Show first 20
            print(f"{i+1:3d}. {value:.4f}")
        if len(similarities) > 20:
            print(f"... and {len(similarities) - 20} more values")
    
    return 0

if __name__ == "__main__":
    exit(main())
