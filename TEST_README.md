# RAG System Test Scripts

This directory contains comprehensive test scripts for evaluating the RAG systems in `main.py` and `main2.py`.

## Test Scripts Overview

### 1. `test_main.py` - Vanilla RAG System Tester
Tests the simple vanilla RAG system from `main.py` which uses:
- Basic ChromaDB similarity search
- Simple document retrieval
- Standard RAG pipeline

### 2. `test_main2.py` - Advanced RAG System Tester  
Tests the advanced RAG system from `main2.py` which uses:
- RRF (Reciprocal Rank Fusion) with 4 retrievers
- Cross-encoder re-ranking
- Language detection (English/Bangla)
- Advanced document filtering

## Prerequisites

Before running the tests, ensure you have:

1. **ChromaDB Index**: Run `docadd_text.py` to build the vector database
2. **Ollama**: Make sure Ollama is running with the required model (`llama3.2:3b`)
3. **Test Data**: Place JSON test files in the `dataqa/` directory
4. **Dependencies**: Install required packages:
   ```bash
   pip install bert-score sentence-transformers rank-bm25 scipy
   ```

## Usage

### Basic Testing

```bash
# Test vanilla RAG system
python test_main.py

# Test advanced RAG system  
python test_main.py
```

### Advanced Options

```bash
# Verbose output (shows detailed progress)
python test_main.py --verbose
python test_main2.py --verbose

# Save detailed results to JSON files
python test_main.py --save
python test_main2.py --save

# Compare advanced RAG with vanilla RAG results
python test_main2.py --compare
```

### Complete Testing Workflow

1. **Test Vanilla RAG**:
   ```bash
   python test_main.py --save --verbose
   ```

2. **Test Advanced RAG**:
   ```bash
   python test_main2.py --save --verbose --compare
   ```

## Test Data Format

The test scripts expect JSON files in the `dataqa/` directory with the following format:

```json
[
  {
    "question": "What documents do I need for a bank account?",
    "answer": "You need your national ID, passport photos, and proof of address."
  },
  {
    "query": "How to apply for a loan?",
    "reference_answer": "Visit any branch with required documents and fill out the application form."
  }
]
```

## Output Files

### Test Results
- `{filename}_vanilla_rag_test_results.json` - Vanilla RAG results
- `{filename}_advanced_rag_test_results.json` - Advanced RAG results

### Metrics Included
- **Performance**: Processing time, document retrieval stats
- **Quality**: BERTScore F1, Precision, Recall, Composite scores
- **Language Analysis**: Language detection accuracy (for advanced RAG)
- **Advanced Features**: Cross-encoder usage, RRF fusion stats

## Key Differences Between Systems

| Feature | Vanilla RAG (main.py) | Advanced RAG (main2.py) |
|---------|----------------------|-------------------------|
| Retrieval | Single ChromaDB search | RRF Fusion (4 retrievers) |
| Re-ranking | None | Cross-encoder re-ranking |
| Language Support | English only | English + Bangla |
| Document Filtering | Basic | Advanced form field filtering |
| Response Quality | Standard | Language-specific prompts |

## Troubleshooting

### Common Issues

1. **ChromaDB not found**: Run `docadd_text.py` first to build the index
2. **Ollama connection error**: Ensure Ollama is running with `llama3.2:3b` model
3. **BERTScore errors**: Install with `pip install bert-score`
4. **Memory issues**: Reduce batch size or use smaller test datasets

### Performance Tips

- Use `--verbose` only for debugging
- Run tests on smaller datasets first
- Ensure sufficient RAM for advanced RAG system
- Close other applications during testing

## Example Output

```
================================================================================
VANILLA RAG SYSTEM TEST RESULTS
================================================================================

📊 Test Summary:
  Total Tests: 50
  Successful: 48
  Failed: 2
  Success Rate: 96.0%

⚡ Performance Metrics:
  Average Processing Time: 2.34s
  Max Processing Time: 5.67s
  Min Processing Time: 1.23s
  Average Documents Retrieved: 2.1

🎯 Quality Metrics (BERTScore):
  Average BERTScore F1: 0.7845
  Average BERTScore Precision: 0.8123
  Average BERTScore Recall: 0.7567
  Average Composite Score: 0.7234
```

## Comparison Mode

When using `--compare`, the advanced RAG tester will automatically compare results with the vanilla RAG system:

```
================================================================================
COMPARISON: ADVANCED RAG vs VANILLA RAG
================================================================================

📊 Performance Comparison:
  Processing Time:
    Advanced RAG: 3.45s
    Vanilla RAG:  2.34s

🎯 Quality Comparison (BERTScore):
  F1 Score:
    Advanced RAG: 0.8234
    Vanilla RAG:  0.7845

📈 Improvement: +5.0% in BERTScore F1
```

This helps you understand the trade-offs between performance and quality when choosing between the two RAG systems.
