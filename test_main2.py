#!/usr/bin/env python3
"""
Test Script for main2.py (Advanced RAG System with RRF Fusion)
==============================================================

This script tests the advanced RAG system from main2.py by:
1. Loading test queries from JSON files
2. Running the advanced RAG system on each query
3. Evaluating responses using BERTScore
4. Generating comprehensive test reports with advanced metrics

Usage:
  python test_main2.py              # Test with default settings
  python test_main2.py --verbose    # Show detailed output
  python test_main2.py --save       # Save detailed results to file
  python test_main2.py --compare    # Compare with vanilla RAG results
"""

import os
import json
import time
import argparse
import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Import the advanced RAG system from main2.py
import sys
sys.path.append('.')
from main2 import CoreRAGSystem

# Import BERTScore evaluator from evalbertscore.py
from evalbertscore import BERTScoreEvaluator

class AdvancedRAGTester:
    """Test suite for the advanced RAG system with RRF fusion"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.rag_system = None
        self.evaluator = None
        self.test_results = []
        
        print("[INFO] Initializing Advanced RAG Tester...")
        self._init_components()
    
    def _init_components(self):
        """Initialize RAG system and evaluator"""
        try:
            # Initialize advanced RAG system
            print("[INFO] Loading Advanced RAG system with RRF fusion...")
            self.rag_system = CoreRAGSystem()
            print("[INFO] ✅ Advanced RAG system loaded successfully")
            
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
        """Run a single test query through the advanced RAG system"""
        start_time = time.time()
        
        try:
            # Process query through advanced RAG
            result = self.rag_system.process_query_sync(query)
            
            processing_time = time.time() - start_time
            
            # Extract response and metadata
            response = result.get("response", "")
            sources = result.get("sources", [])
            contexts = result.get("contexts", [])
            detected_language = result.get("detected_language", "unknown")
            language_confidence = result.get("language_confidence", 0.0)
            documents_found = result.get("documents_found", 0)
            documents_used = result.get("documents_used", 0)
            retrieval_method = result.get("retrieval_method", "unknown")
            cross_encoder_used = result.get("cross_encoder_used", False)
            
            test_result = {
                "query": query,
                "response": response,
                "reference_answer": reference_answer,
                "sources": sources,
                "contexts": contexts,
                "detected_language": detected_language,
                "language_confidence": language_confidence,
                "documents_found": documents_found,
                "documents_used": documents_used,
                "retrieval_method": retrieval_method,
                "cross_encoder_used": cross_encoder_used,
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
                "detected_language": "unknown",
                "language_confidence": 0.0,
                "documents_found": 0,
                "documents_used": 0,
                "retrieval_method": "error",
                "cross_encoder_used": False,
                "processing_time": round(time.time() - start_time, 2),
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def run_batch_tests(self, test_data: List[Dict]) -> List[Dict]:
        """Run batch tests on all test data"""
        print(f"\n🚀 Running {len(test_data)} tests on Advanced RAG system...")
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
                print(f"  Language: {result['detected_language']} (confidence: {result['language_confidence']:.2f})")
                print(f"  Processing time: {result['processing_time']}s")
                print(f"  Documents found: {result['documents_found']}, used: {result['documents_used']}")
                print(f"  Retrieval method: {result['retrieval_method']}")
                print(f"  Cross-encoder: {'✅' if result['cross_encoder_used'] else '❌'}")
                
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
        docs_found_list = [r.get("documents_found", 0) for r in results if "error" not in r]
        docs_used_list = [r.get("documents_used", 0) for r in results if "error" not in r]
        avg_docs_found = sum(docs_found_list) / len(docs_found_list) if docs_found_list else 0
        avg_docs_used = sum(docs_used_list) / len(docs_used_list) if docs_used_list else 0
        max_docs_found = max(docs_found_list) if docs_found_list else 0
        max_docs_used = max(docs_used_list) if docs_used_list else 0
        
        # Language detection statistics
        language_counts = {}
        language_confidences = []
        for r in results:
            if "error" not in r:
                lang = r.get("detected_language", "unknown")
                language_counts[lang] = language_counts.get(lang, 0) + 1
                confidence = r.get("language_confidence", 0.0)
                if confidence > 0:
                    language_confidences.append(confidence)
        
        avg_language_confidence = sum(language_confidences) / len(language_confidences) if language_confidences else 0
        
        # Cross-encoder usage statistics
        cross_encoder_used_count = len([r for r in results if r.get("cross_encoder_used", False)])
        cross_encoder_usage_rate = (cross_encoder_used_count / total_tests * 100) if total_tests > 0 else 0
        
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
                "avg_docs_found": round(avg_docs_found, 1),
                "avg_docs_used": round(avg_docs_used, 1),
                "max_docs_found": max_docs_found,
                "max_docs_used": max_docs_used
            },
            "language_analysis": {
                "language_distribution": language_counts,
                "avg_language_confidence": round(avg_language_confidence, 3),
                "total_languages_detected": len(language_counts)
            },
            "advanced_features": {
                "cross_encoder_used_count": cross_encoder_used_count,
                "cross_encoder_usage_rate": round(cross_encoder_usage_rate, 1),
                "retrieval_method": "rrf_fusion"
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
        print("ADVANCED RAG SYSTEM TEST RESULTS")
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
        print(f"  Average Documents Found: {perf.get('avg_docs_found', 0)}")
        print(f"  Average Documents Used: {perf.get('avg_docs_used', 0)}")
        print(f"  Max Documents Found: {perf.get('max_docs_found', 0)}")
        print(f"  Max Documents Used: {perf.get('max_docs_used', 0)}")
        
        # Language Analysis
        lang_analysis = stats.get("language_analysis", {})
        print(f"\n🌐 Language Analysis:")
        lang_dist = lang_analysis.get("language_distribution", {})
        for lang, count in lang_dist.items():
            percentage = (count / summary.get('total_tests', 1) * 100) if summary.get('total_tests', 0) > 0 else 0
            print(f"  {lang.title()}: {count} ({percentage:.1f}%)")
        print(f"  Average Language Confidence: {lang_analysis.get('avg_language_confidence', 0):.3f}")
        print(f"  Total Languages Detected: {lang_analysis.get('total_languages_detected', 0)}")
        
        # Advanced Features
        advanced = stats.get("advanced_features", {})
        print(f"\n🔬 Advanced Features:")
        print(f"  Retrieval Method: {advanced.get('retrieval_method', 'unknown')}")
        print(f"  Cross-Encoder Used: {advanced.get('cross_encoder_used_count', 0)} times")
        print(f"  Cross-Encoder Usage Rate: {advanced.get('cross_encoder_usage_rate', 0):.1f}%")
        
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
        print(f"  RAG System: Advanced RAG with RRF Fusion (main2.py)")
        print(f"  Embedding Model: BAAI/bge-m3")
        print(f"  Cross-Encoder: BAAI/bge-reranker-v2-m3")
        print(f"  LLM: Llama3.2:3b (Ollama)")
        print(f"  Vector Store: ChromaDB")
        print(f"  Retrieval Methods: RRF Fusion (4 retrievers)")
        print(f"  Languages: English + বাংলা (Bangla)")
    
    def save_results(self, results: List[Dict], stats: Dict, test_file: str):
        """Save detailed results to JSON file"""
        base_name = os.path.splitext(os.path.basename(test_file))[0]
        results_file = f"{base_name}_advanced_rag_test_results.json"
        
        try:
            output_data = {
                "test_metadata": {
                    "test_file": test_file,
                    "rag_system": "Advanced RAG with RRF Fusion (main2.py)",
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
    
    def compare_with_vanilla(self, test_file: str):
        """Compare results with vanilla RAG system if results exist"""
        base_name = os.path.splitext(os.path.basename(test_file))[0]
        vanilla_results_file = f"{base_name}_vanilla_rag_test_results.json"
        
        if not os.path.exists(vanilla_results_file):
            print(f"[INFO] No vanilla RAG results found at {vanilla_results_file}")
            print("[INFO] Run test_main.py first to generate comparison data")
            return
        
        try:
            with open(vanilla_results_file, 'r', encoding='utf-8') as f:
                vanilla_data = json.load(f)
            
            vanilla_stats = vanilla_data.get("statistics", {})
            
            print(f"\n" + "=" * 80)
            print("COMPARISON: ADVANCED RAG vs VANILLA RAG")
            print("=" * 80)
            
            # Compare key metrics
            print(f"\n📊 Performance Comparison:")
            print(f"  Processing Time:")
            print(f"    Advanced RAG: {self.current_stats.get('performance', {}).get('avg_processing_time', 0)}s")
            print(f"    Vanilla RAG:  {vanilla_stats.get('performance', {}).get('avg_processing_time', 0)}s")
            
            print(f"\n  Document Retrieval:")
            print(f"    Advanced RAG: {self.current_stats.get('performance', {}).get('avg_docs_used', 0)} docs used")
            print(f"    Vanilla RAG:  {vanilla_stats.get('performance', {}).get('avg_docs_retrieved', 0)} docs retrieved")
            
            print(f"\n🎯 Quality Comparison (BERTScore):")
            print(f"  F1 Score:")
            print(f"    Advanced RAG: {self.current_stats.get('quality_metrics', {}).get('avg_bert_score_f1', 0):.4f}")
            print(f"    Vanilla RAG:  {vanilla_stats.get('quality_metrics', {}).get('avg_bert_score_f1', 0):.4f}")
            
            print(f"  Composite Score:")
            print(f"    Advanced RAG: {self.current_stats.get('quality_metrics', {}).get('avg_composite_bert_score', 0):.4f}")
            print(f"    Vanilla RAG:  {vanilla_stats.get('quality_metrics', {}).get('avg_composite_bert_score', 0):.4f}")
            
            # Calculate improvements
            adv_f1 = self.current_stats.get('quality_metrics', {}).get('avg_bert_score_f1', 0)
            van_f1 = vanilla_stats.get('quality_metrics', {}).get('avg_bert_score_f1', 0)
            if van_f1 > 0:
                f1_improvement = ((adv_f1 - van_f1) / van_f1) * 100
                print(f"\n📈 Improvement: {f1_improvement:+.1f}% in BERTScore F1")
            
        except Exception as e:
            print(f"[WARNING] Failed to load vanilla results for comparison: {e}")

def main():
    """Main function for advanced RAG testing"""
    parser = argparse.ArgumentParser(
        description="Test Advanced RAG System with RRF Fusion (main2.py)",
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
    
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Compare results with vanilla RAG system"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ADVANCED RAG SYSTEM TESTER (RRF Fusion)")
    print("=" * 80)
    
    try:
        # Initialize tester
        tester = AdvancedRAGTester(verbose=args.verbose)
        
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
        tester.current_stats = stats  # Store for comparison
        
        # Print results
        tester.print_results(results, stats)
        
        # Save results if requested
        if args.save:
            tester.save_results(results, stats, test_file)
        
        # Compare with vanilla RAG if requested
        if args.compare:
            tester.compare_with_vanilla(test_file)
        
        print("\n" + "=" * 80)
        print("ADVANCED RAG SYSTEM TESTING COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
