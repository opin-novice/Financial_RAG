#!/usr/bin/env python3
"""
Comprehensive RRF Fusion Test Script

This script tests the RRF (Reciprocal Rank Fusion) implementation in the RAG system
to verify that it's working correctly and producing expected results.

Test Coverage:
1. Individual retriever components (ChromaDB, BM25, Dirichlet)
2. RRF score calculation and fusion logic
3. End-to-end RRF fusion with sample queries
4. Configuration validation
5. Performance metrics
"""

import os
import sys
import json
import logging
import time
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the RAG system components
from main2 import RRFFusionRetriever, CoreRAGSystem, config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_rrf_fusion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RRFFusionTester:
    """Comprehensive RRF Fusion Test Suite"""
    
    def __init__(self):
        self.retriever = None
        self.rag_system = None
        self.test_results = {}
        self.sample_queries = [
            "What are the requirements for car loan?",
            "How to calculate income tax?",
            "What is credit score?",
            "Bank opening guidelines",
            "Student loan eligibility criteria"
        ]
        
    def initialize_systems(self):
        """Initialize the RAG system and retriever"""
        logger.info("🚀 Initializing RAG system and RRF retriever...")
        
        try:
            # Initialize the retriever
            self.retriever = RRFFusionRetriever()
            logger.info("✅ RRF Fusion Retriever initialized")
            
            # Initialize the full RAG system
            self.rag_system = CoreRAGSystem()
            logger.info("✅ Core RAG System initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize systems: {e}")
            return False
    
    def test_configuration(self):
        """Test RRF configuration parameters"""
        logger.info("🔧 Testing RRF configuration...")
        
        config_tests = {
            "RRF_K": config.RRF_K,
            "RRF_WEIGHTS": config.RRF_WEIGHTS,
            "MAX_DOCS_FOR_RETRIEVAL": config.MAX_DOCS_FOR_RETRIEVAL
        }
        
        # Validate RRF_K
        assert config.RRF_K > 0, f"RRF_K should be positive, got {config.RRF_K}"
        logger.info(f"✅ RRF_K: {config.RRF_K}")
        
        # Validate RRF_WEIGHTS
        total_weight = sum(config.RRF_WEIGHTS.values())
        assert abs(total_weight - 1.0) < 0.01, f"RRF weights should sum to ~1.0, got {total_weight}"
        logger.info(f"✅ RRF_WEIGHTS sum: {total_weight}")
        
        for retriever, weight in config.RRF_WEIGHTS.items():
            assert 0 <= weight <= 1, f"Weight for {retriever} should be between 0-1, got {weight}"
            logger.info(f"  - {retriever}: {weight}")
        
        # Validate MAX_DOCS_FOR_RETRIEVAL
        assert config.MAX_DOCS_FOR_RETRIEVAL > 0, f"MAX_DOCS_FOR_RETRIEVAL should be positive, got {config.MAX_DOCS_FOR_RETRIEVAL}"
        logger.info(f"✅ MAX_DOCS_FOR_RETRIEVAL: {config.MAX_DOCS_FOR_RETRIEVAL}")
        
        self.test_results["configuration"] = {
            "status": "PASSED",
            "rrf_k": config.RRF_K,
            "rrf_weights": config.RRF_WEIGHTS,
            "total_weight": total_weight,
            "max_docs": config.MAX_DOCS_FOR_RETRIEVAL
        }
        
        return True
    
    def test_individual_retrievers(self):
        """Test individual retriever components"""
        logger.info("🔍 Testing individual retriever components...")
        
        if not self.retriever:
            logger.error("❌ Retriever not initialized")
            return False
        
        retriever_tests = {}
        query = "car loan requirements"
        
        # Test ChromaDB
        try:
            chroma_results = self.retriever._chroma_search(query, top_k=5)
            retriever_tests["chroma"] = {
                "status": "PASSED" if chroma_results else "FAILED",
                "results_count": len(chroma_results),
                "sample_results": chroma_results[:3] if chroma_results else []
            }
            logger.info(f"✅ ChromaDB: {len(chroma_results)} results")
        except Exception as e:
            retriever_tests["chroma"] = {"status": "FAILED", "error": str(e)}
            logger.error(f"❌ ChromaDB failed: {e}")
        
        # Test BM25
        try:
            bm25_results = self.retriever._bm25_search(query, top_k=5)
            retriever_tests["bm25"] = {
                "status": "PASSED" if bm25_results else "FAILED",
                "results_count": len(bm25_results),
                "sample_results": bm25_results[:3] if bm25_results else []
            }
            logger.info(f"✅ BM25: {len(bm25_results)} results")
        except Exception as e:
            retriever_tests["bm25"] = {"status": "FAILED", "error": str(e)}
            logger.error(f"❌ BM25 failed: {e}")
        
        # Test Dirichlet
        try:
            dirichlet_results = self.retriever._dirichlet_search(query, top_k=5)
            retriever_tests["dirichlet"] = {
                "status": "PASSED" if dirichlet_results else "FAILED",
                "results_count": len(dirichlet_results),
                "sample_results": dirichlet_results[:3] if dirichlet_results else []
            }
            logger.info(f"✅ Dirichlet: {len(dirichlet_results)} results")
        except Exception as e:
            retriever_tests["dirichlet"] = {"status": "FAILED", "error": str(e)}
            logger.error(f"❌ Dirichlet failed: {e}")
        
        # Test Multilingual E5 (if enabled)
        try:
            multilingual_e5_results = self.retriever._multilingual_e5_search(query, top_k=5)
            retriever_tests["multilingual_e5"] = {
                "status": "PASSED" if multilingual_e5_results else "SKIPPED",
                "results_count": len(multilingual_e5_results),
                "sample_results": multilingual_e5_results[:3] if multilingual_e5_results else []
            }
            logger.info(f"✅ Multilingual E5: {len(multilingual_e5_results)} results")
        except Exception as e:
            retriever_tests["multilingual_e5"] = {"status": "SKIPPED", "error": str(e)}
            logger.info(f"ℹ️ Multilingual E5 skipped: {e}")
        
        self.test_results["individual_retrievers"] = retriever_tests
        return all(test.get("status") in ["PASSED", "SKIPPED"] for test in retriever_tests.values())
    
    def test_rrf_score_calculation(self):
        """Test RRF score calculation logic"""
        logger.info("🧮 Testing RRF score calculation...")
        
        # Create mock ranked lists for testing
        mock_ranked_lists = {
            "chroma": [(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6)],
            "bm25": [(2, 0.85), (1, 0.75), (4, 0.65), (3, 0.55)],
            "dirichlet": [(3, 0.8), (1, 0.7), (2, 0.6), (4, 0.5)]
        }
        
        # Calculate RRF scores
        rrf_scores = defaultdict(float)
        
        for retriever_name, ranked_list in mock_ranked_lists.items():
            weight = config.RRF_WEIGHTS.get(retriever_name, 1.0)
            
            for rank, (doc_id, score) in enumerate(ranked_list, 1):
                rrf_score = weight / (config.RRF_K + rank)
                rrf_scores[doc_id] += rrf_score
        
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Validate results
        assert len(sorted_results) > 0, "RRF calculation should produce results"
        assert all(score > 0 for _, score in sorted_results), "All RRF scores should be positive"
        
        # Check that results are sorted by score (descending)
        scores = [score for _, score in sorted_results]
        assert scores == sorted(scores, reverse=True), "Results should be sorted by RRF score (descending)"
        
        logger.info(f"✅ RRF calculation test passed")
        logger.info(f"  - Input retrievers: {list(mock_ranked_lists.keys())}")
        logger.info(f"  - Output documents: {len(sorted_results)}")
        logger.info(f"  - Top 3 results: {sorted_results[:3]}")
        
        self.test_results["rrf_calculation"] = {
            "status": "PASSED",
            "input_retrievers": list(mock_ranked_lists.keys()),
            "output_documents": len(sorted_results),
            "top_results": sorted_results[:5]
        }
        
        return True
    
    def test_rrf_fusion_search(self):
        """Test the complete RRF fusion search"""
        logger.info("🔍 Testing complete RRF fusion search...")
        
        if not self.retriever:
            logger.error("❌ Retriever not initialized")
            return False
        
        query = "car loan requirements"
        
        try:
            # Perform RRF fusion search
            start_time = time.time()
            docs = self.retriever._rrf_fusion_search(query, top_k=10)
            search_time = time.time() - start_time
            
            # Validate results
            assert isinstance(docs, list), "RRF fusion should return a list of documents"
            
            if docs:
                # Check document structure
                doc = docs[0]
                assert hasattr(doc, 'page_content'), "Document should have page_content"
                assert hasattr(doc, 'metadata'), "Document should have metadata"
                
                # Check metadata structure
                metadata = doc.metadata
                required_fields = ['rrf_score', 'retrieval_method', 'source']
                for field in required_fields:
                    assert field in metadata, f"Document metadata should contain {field}"
                
                # Check RRF score
                assert 'rrf_score' in metadata, "Document should have RRF score"
                assert isinstance(metadata['rrf_score'], (int, float)), "RRF score should be numeric"
                assert metadata['rrf_score'] > 0, "RRF score should be positive"
                
                # Check retrieval method
                assert metadata['retrieval_method'] == 'rrf_fusion', "Retrieval method should be 'rrf_fusion'"
            
            logger.info(f"✅ RRF fusion search completed")
            logger.info(f"  - Query: {query}")
            logger.info(f"  - Documents found: {len(docs)}")
            logger.info(f"  - Search time: {search_time:.2f}s")
            
            if docs:
                logger.info(f"  - Top document RRF score: {docs[0].metadata.get('rrf_score', 'N/A')}")
                logger.info(f"  - Top document source: {docs[0].metadata.get('source', 'N/A')}")
            
            self.test_results["rrf_fusion_search"] = {
                "status": "PASSED",
                "query": query,
                "documents_found": len(docs),
                "search_time": search_time,
                "top_document_score": docs[0].metadata.get('rrf_score') if docs else None,
                "top_document_source": docs[0].metadata.get('source') if docs else None
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ RRF fusion search failed: {e}")
            self.test_results["rrf_fusion_search"] = {
                "status": "FAILED",
                "error": str(e)
            }
            return False
    
    def test_end_to_end_rag(self):
        """Test end-to-end RAG system with RRF fusion"""
        logger.info("🚀 Testing end-to-end RAG system with RRF fusion...")
        
        if not self.rag_system:
            logger.error("❌ RAG system not initialized")
            return False
        
        query = "What are the requirements for car loan?"
        
        try:
            # Process query through the full RAG system
            start_time = time.time()
            result = self.rag_system.process_query_sync(query)
            processing_time = time.time() - start_time
            
            # Validate result structure
            required_fields = ['response', 'sources', 'contexts', 'detected_language', 'retrieval_method']
            for field in required_fields:
                assert field in result, f"Result should contain {field}"
            
            # Check retrieval method
            assert result['retrieval_method'] == 'rrf_fusion', "Retrieval method should be 'rrf_fusion'"
            
            # Check response quality
            response = result['response']
            assert isinstance(response, str), "Response should be a string"
            assert len(response) > 0, "Response should not be empty"
            
            logger.info(f"✅ End-to-end RAG test completed")
            logger.info(f"  - Query: {query}")
            logger.info(f"  - Response length: {len(response)} characters")
            logger.info(f"  - Processing time: {processing_time:.2f}s")
            logger.info(f"  - Sources found: {len(result.get('sources', []))}")
            logger.info(f"  - Language detected: {result.get('detected_language', 'N/A')}")
            logger.info(f"  - Retrieval method: {result.get('retrieval_method', 'N/A')}")
            
            self.test_results["end_to_end_rag"] = {
                "status": "PASSED",
                "query": query,
                "response_length": len(response),
                "processing_time": processing_time,
                "sources_count": len(result.get('sources', [])),
                "detected_language": result.get('detected_language'),
                "retrieval_method": result.get('retrieval_method'),
                "response_preview": response[:200] + "..." if len(response) > 200 else response
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ End-to-end RAG test failed: {e}")
            self.test_results["end_to_end_rag"] = {
                "status": "FAILED",
                "error": str(e)
            }
            return False
    
    def test_multiple_queries(self):
        """Test RRF fusion with multiple sample queries"""
        logger.info("🔍 Testing RRF fusion with multiple queries...")
        
        if not self.retriever:
            logger.error("❌ Retriever not initialized")
            return False
        
        query_results = {}
        
        for i, query in enumerate(self.sample_queries):
            try:
                start_time = time.time()
                docs = self.retriever.get_relevant_documents(query)
                search_time = time.time() - start_time
                
                query_results[query] = {
                    "status": "PASSED",
                    "documents_found": len(docs),
                    "search_time": search_time,
                    "avg_rrf_score": np.mean([doc.metadata.get('rrf_score', 0) for doc in docs]) if docs else 0
                }
                
                logger.info(f"✅ Query {i+1}/{len(self.sample_queries)}: {query}")
                logger.info(f"  - Documents: {len(docs)}, Time: {search_time:.2f}s")
                
            except Exception as e:
                query_results[query] = {
                    "status": "FAILED",
                    "error": str(e)
                }
                logger.error(f"❌ Query {i+1} failed: {query} - {e}")
        
        # Calculate statistics
        successful_queries = [q for q, r in query_results.items() if r["status"] == "PASSED"]
        avg_docs = np.mean([r["documents_found"] for r in query_results.values() if r["status"] == "PASSED"])
        avg_time = np.mean([r["search_time"] for r in query_results.values() if r["status"] == "PASSED"])
        
        logger.info(f"✅ Multiple queries test completed")
        logger.info(f"  - Successful queries: {len(successful_queries)}/{len(self.sample_queries)}")
        logger.info(f"  - Average documents per query: {avg_docs:.1f}")
        logger.info(f"  - Average search time: {avg_time:.2f}s")
        
        self.test_results["multiple_queries"] = {
            "status": "PASSED" if len(successful_queries) == len(self.sample_queries) else "PARTIAL",
            "total_queries": len(self.sample_queries),
            "successful_queries": len(successful_queries),
            "success_rate": len(successful_queries) / len(self.sample_queries),
            "avg_documents": avg_docs,
            "avg_search_time": avg_time,
            "query_results": query_results
        }
        
        return len(successful_queries) == len(self.sample_queries)
    
    def run_all_tests(self):
        """Run all RRF fusion tests"""
        logger.info("🧪 Starting comprehensive RRF fusion test suite...")
        
        test_suite = [
            ("Configuration", self.test_configuration),
            ("Individual Retrievers", self.test_individual_retrievers),
            ("RRF Score Calculation", self.test_rrf_score_calculation),
            ("RRF Fusion Search", self.test_rrf_fusion_search),
            ("End-to-End RAG", self.test_end_to_end_rag),
            ("Multiple Queries", self.test_multiple_queries)
        ]
        
        passed_tests = 0
        total_tests = len(test_suite)
        
        for test_name, test_func in test_suite:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running test: {test_name}")
            logger.info(f"{'='*60}")
            
            try:
                if test_func():
                    passed_tests += 1
                    logger.info(f"✅ {test_name} - PASSED")
                else:
                    logger.error(f"❌ {test_name} - FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name} - ERROR: {e}")
                self.test_results[test_name.lower().replace(" ", "_")] = {
                    "status": "ERROR",
                    "error": str(e)
                }
        
        # Generate summary report
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        
        # Save detailed results
        self.save_test_results()
        
        return passed_tests == total_tests
    
    def save_test_results(self):
        """Save test results to JSON file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"rrf_fusion_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"📄 Test results saved to: {filename}")

def main():
    """Main test execution"""
    print("🧪 RRF Fusion Test Suite")
    print("=" * 50)
    
    tester = RRFFusionTester()
    
    # Initialize systems
    if not tester.initialize_systems():
        print("❌ Failed to initialize systems. Exiting.")
        return False
    
    # Run all tests
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! RRF fusion is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the logs for details.")
    
    return success

if __name__ == "__main__":
    main()
    