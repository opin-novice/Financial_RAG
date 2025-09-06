#!/usr/bin/env python3
"""
Test Script for main.py (Vanilla RAG System)
============================================

This script tests the vanilla RAG system from main.py by:
1. Loading test queries from JSON files
2. Running the RAG system on each query
3. Evaluating responses using BERTScore
4. Generating comprehensive test reports

Usage:
  python test_main.py              # Test with default settings
  python test_main.py --verbose    # Show detailed output
  python test_main.py --save       # Save detailed results to file
"""

import os
import json
import time
import argparse
import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Import the vanilla RAG system from main.py
import sys
sys.path.append('.')
from main import VanillaRAGSystem

# Import BERTScore evaluator from evalbertscore.py
from evalbertscore import BERTScoreEvaluator

class VanillaRAGTester:
    """Test suite for the vanilla RAG system"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.rag_system = None
        self.evaluator = None
        self.test_results = []
        
        print("[INFO] Initializing Vanilla RAG Tester...")
        self._init_components()
    
    def _init_components(self):
        """Initialize RAG system and evaluator"""
        try:
            # Initialize vanilla RAG system
            print("[INFO] Loading Vanilla RAG system...")
            self.rag_system = VanillaRAGSystem()
            print("[INFO] ✅ Vanilla RAG system loaded successfully")
            
            # Initialize BERTScore evaluator
            print("[INFO] Loading BERTScore evaluator...")
            self.evaluator = BERTScoreEvaluator(verbose=self.verbose)
            print("[INFO] ✅ BERTScore evaluator loaded successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize components: {e}")
            raise e
    
    def find_test_files(self, directory: str = "dataqa") -> List[str]:
        """Find JSON test files in the dataqa directory"""
        json_files = []
        
        if os.path.exists(directory):
            json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
            json_files = [os.path.join(directory, f) for f in json_files]
        
        return sorted(json_files)
    
    def select_test_file(self, test_files: List[str]) -> Optional[str]:
        """Allow user to select which test file to use"""
        if not test_files:
            print("[ERROR] No JSON test files found in dataqa directory")
            return None
        
        if len(test_files) == 1:
            print(f"[INFO] Found test file: {test_files[0]}")
            return test_files[0]
        
        print("\n📁 Available test files:")
        for i, file_path in enumerate(test_files, 1):
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            print(f"  {i}. {file_path} ({file_size:,} bytes)")
        
        while True:
            try:
                choice = input(f"\nSelect file (1-{len(test_files)}) or press Enter for first file: ").strip()
                
                if not choice:
                    return test_files[0]
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(test_files):
                    return test_files[choice_idx]
                else:
                    print(f"Please enter a number between 1 and {len(test_files)}")
            
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\n[INFO] Testing cancelled")
                return None
    
    def load_test_data(self, test_file: str) -> List[Dict]:
        """Load test data from JSON file"""
        print(f"[INFO] Loading test data from {test_file}...")
        
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            print(f"[INFO] Successfully loaded {len(test_data)} test items")
            return test_data
            
        except Exception as e:
            print(f"[ERROR] Failed to load test data: {e}")
            return []
    
    def run_single_test(self, query: str, reference_answer: str = None) -> Dict:
        """Run a single test query through the vanilla RAG system"""
        start_time = time.time()
        
        try:
            # Process query through vanilla RAG
            result = self.rag_system.process_query(query)
            
            processing_time = time.time() - start_time
            
            # Extract response and metadata
            response = result.get("response", "")
            sources = result.get("sources", [])
            contexts = result.get("contexts", [])
            num_docs = result.get("num_docs", 0)
            
            test_result = {
                "query": query,
                "response": response,
                "reference_answer": reference_answer,
                "sources": sources,
                "contexts": contexts,
                "num_docs": num_docs,
                "processing_time": round(processing_time, 2),
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Evaluate with BERTScore if reference answer is provided
            if reference_answer and response:
                try:
                    bert_scores = self.evaluator.evaluate_answer(
                        predicted=response,
                        reference=reference_answer,
                        context=contexts if contexts else None
                    )
                    test_result["bert_scores"] = bert_scores
                except Exception as e:
                    print(f"[WARNING] BERTScore evaluation failed: {e}")
                    test_result["bert_scores"] = {
                        "bert_score_precision": 0.0,
                        "bert_score_recall": 0.0,
                        "bert_score_f1": 0.0,
                        "composite_bert_score": 0.0
                    }
            
            return test_result
            
        except Exception as e:
            print(f"[ERROR] Test failed for query '{query}': {e}")
            return {
                "query": query,
                "response": f"Error: {e}",
                "reference_answer": reference_answer,
                "sources": [],
                "contexts": [],
                "num_docs": 0,
                "processing_time": round(time.time() - start_time, 2),
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def run_batch_tests(self, test_data: List[Dict]) -> List[Dict]:
        """Run batch tests on all test data"""
        print(f"\n🚀 Running {len(test_data)} tests on Vanilla RAG system...")
        print("=" * 80)
        
        results = []
        
        for i, test_item in enumerate(test_data, 1):
            query = test_item.get("question", test_item.get("query", ""))
            reference = test_item.get("answer", test_item.get("reference_answer", ""))
            
            if not query:
                print(f"[WARNING] Skipping test {i}: No query found")
                continue
            
            print(f"\n📝 Test {i}/{len(test_data)}: {query[:100]}{'...' if len(query) > 100 else ''}")
            
            result = self.run_single_test(query, reference)
            results.append(result)
            
            # Show progress
            if self.verbose:
                print(f"  Response: {result['response'][:200]}{'...' if len(result['response']) > 200 else ''}")
                print(f"  Processing time: {result['processing_time']}s")
                print(f"  Documents retrieved: {result['num_docs']}")
                
                if "bert_scores" in result:
                    scores = result["bert_scores"]
                    print(f"  BERTScore F1: {scores.get('bert_score_f1', 0):.3f}")
            
            # Progress indicator
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(test_data)} tests completed")
        
        print(f"\n✅ Batch testing completed: {len(results)} tests")
        return results
    
    def calculate_statistics(self, results: List[Dict]) -> Dict:
        """Calculate comprehensive statistics from test results"""
        if not results:
            return {}
        
        # Basic statistics
        total_tests = len(results)
        successful_tests = len([r for r in results if "error" not in r])
        failed_tests = total_tests - successful_tests
        
        # Processing time statistics
        processing_times = [r.get("processing_time", 0) for r in results if "error" not in r]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        max_processing_time = max(processing_times) if processing_times else 0
        min_processing_time = min(processing_times) if processing_times else 0
        
        # Document retrieval statistics
        num_docs_list = [r.get("num_docs", 0) for r in results if "error" not in r]
        avg_docs_retrieved = sum(num_docs_list) / len(num_docs_list) if num_docs_list else 0
        max_docs_retrieved = max(num_docs_list) if num_docs_list else 0
        
        # BERTScore statistics
        bert_scores = [r.get("bert_scores", {}) for r in results if "bert_scores" in r]
        
        avg_bert_f1 = 0
        avg_bert_precision = 0
        avg_bert_recall = 0
        avg_composite_score = 0
        
        if bert_scores:
            avg_bert_f1 = sum(s.get("bert_score_f1", 0) for s in bert_scores) / len(bert_scores)
            avg_bert_precision = sum(s.get("bert_score_precision", 0) for s in bert_scores) / len(bert_scores)
            avg_bert_recall = sum(s.get("bert_score_recall", 0) for s in bert_scores) / len(bert_scores)
            avg_composite_score = sum(s.get("composite_bert_score", 0) for s in bert_scores) / len(bert_scores)
        
        # Response length statistics
        response_lengths = [len(r.get("response", "")) for r in results if "error" not in r]
        avg_response_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
        max_response_length = max(response_lengths) if response_lengths else 0
        min_response_length = min(response_lengths) if response_lengths else 0
        
        return {
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "performance": {
                "avg_processing_time": round(avg_processing_time, 2),
                "max_processing_time": round(max_processing_time, 2),
                "min_processing_time": round(min_processing_time, 2),
                "avg_docs_retrieved": round(avg_docs_retrieved, 1),
                "max_docs_retrieved": max_docs_retrieved
            },
            "quality_metrics": {
                "avg_bert_score_f1": round(avg_bert_f1, 4),
                "avg_bert_score_precision": round(avg_bert_precision, 4),
                "avg_bert_score_recall": round(avg_bert_recall, 4),
                "avg_composite_bert_score": round(avg_composite_score, 4),
                "tests_with_bert_scores": len(bert_scores)
            },
            "response_analysis": {
                "avg_response_length": round(avg_response_length, 1),
                "max_response_length": max_response_length,
                "min_response_length": min_response_length
            }
        }
    
    def print_results(self, results: List[Dict], stats: Dict):
        """Print comprehensive test results"""
        print("\n" + "=" * 80)
        print("VANILLA RAG SYSTEM TEST RESULTS")
        print("=" * 80)
        
        # Test Summary
        summary = stats.get("test_summary", {})
        print(f"\n📊 Test Summary:")
        print(f"  Total Tests: {summary.get('total_tests', 0)}")
        print(f"  Successful: {summary.get('successful_tests', 0)}")
        print(f"  Failed: {summary.get('failed_tests', 0)}")
        print(f"  Success Rate: {summary.get('success_rate', 0):.1f}%")
        
        # Performance Metrics
        perf = stats.get("performance", {})
        print(f"\n⚡ Performance Metrics:")
        print(f"  Average Processing Time: {perf.get('avg_processing_time', 0)}s")
        print(f"  Max Processing Time: {perf.get('max_processing_time', 0)}s")
        print(f"  Min Processing Time: {perf.get('min_processing_time', 0)}s")
        print(f"  Average Documents Retrieved: {perf.get('avg_docs_retrieved', 0)}")
        print(f"  Max Documents Retrieved: {perf.get('max_docs_retrieved', 0)}")
        
        # Quality Metrics
        quality = stats.get("quality_metrics", {})
        print(f"\n🎯 Quality Metrics (BERTScore):")
        print(f"  Average BERTScore F1: {quality.get('avg_bert_score_f1', 0):.4f}")
        print(f"  Average BERTScore Precision: {quality.get('avg_bert_score_precision', 0):.4f}")
        print(f"  Average BERTScore Recall: {quality.get('avg_bert_score_recall', 0):.4f}")
        print(f"  Average Composite Score: {quality.get('avg_composite_bert_score', 0):.4f}")
        print(f"  Tests with BERTScore: {quality.get('tests_with_bert_scores', 0)}")
        
        # Response Analysis
        response = stats.get("response_analysis", {})
        print(f"\n📝 Response Analysis:")
        print(f"  Average Response Length: {response.get('avg_response_length', 0)} characters")
        print(f"  Max Response Length: {response.get('max_response_length', 0)} characters")
        print(f"  Min Response Length: {response.get('min_response_length', 0)} characters")
        
        # System Information
        print(f"\n🔧 System Information:")
        print(f"  RAG System: Vanilla RAG (main.py)")
        print(f"  Embedding Model: BAAI/bge-m3")
        print(f"  LLM: Llama3.2:3b (Ollama)")
        print(f"  Vector Store: ChromaDB")
        print(f"  Retrieval Method: Similarity Search")
    
    def save_results(self, results: List[Dict], stats: Dict, test_file: str):
        """Save detailed results to JSON file"""
        base_name = os.path.splitext(os.path.basename(test_file))[0]
        results_file = f"{base_name}_vanilla_rag_test_results.json"
        
        try:
            output_data = {
                "test_metadata": {
                    "test_file": test_file,
                    "rag_system": "Vanilla RAG (main.py)",
                    "test_timestamp": datetime.datetime.now().isoformat(),
                    "total_tests": len(results)
                },
                "statistics": stats,
                "detailed_results": results
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Detailed results saved to: {results_file}")
            return results_file
            
        except Exception as e:
            print(f"[WARNING] Failed to save results: {e}")
            return None

def main():
    """Main function for vanilla RAG testing"""
    parser = argparse.ArgumentParser(
        description="Test Vanilla RAG System (main.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output during testing"
    )
    
    parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="Save detailed results to JSON file"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("VANILLA RAG SYSTEM TESTER")
    print("=" * 80)
    
    try:
        # Initialize tester
        tester = VanillaRAGTester(verbose=args.verbose)
        
        # Find test files
        test_files = tester.find_test_files()
        
        # Select test file
        test_file = tester.select_test_file(test_files)
        if not test_file:
            return
        
        # Load test data
        test_data = tester.load_test_data(test_file)
        if not test_data:
            print("[ERROR] No valid test data found")
            return
        
        # Run tests
        results = tester.run_batch_tests(test_data)
        
        # Calculate statistics
        stats = tester.calculate_statistics(results)
        
        # Print results
        tester.print_results(results, stats)
        
        # Save results if requested
        if args.save:
            tester.save_results(results, stats, test_file)
        
        print("\n" + "=" * 80)
        print("VANILLA RAG SYSTEM TESTING COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
