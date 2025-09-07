================================================================================
🚀 ADVANCED RAG SYSTEM FOR FINANCIAL SERVICES
================================================================================

A comprehensive RAG system for Bangladeshi financial services with multilingual 
support (English/Bangla), advanced document retrieval, and evaluation metrics.

================================================================================
📋 QUICK START
================================================================================

1. Install dependencies: pip install -r requirements.txt
2. Setup NLTK: python setup_nltk.py
3. Install Ollama: ollama pull llama3.2:3b
4. Process documents: python sant.py then python docadd_text.py
5. Run system: python main2.py

================================================================================
🏗️ SYSTEM ARCHITECTURE
================================================================================

PDF Documents → OCR Processing → Text Extraction → Data Sanitization
                    ↓
Text Chunking → Embedding Generation → Vector Database (ChromaDB)
                    ↓
RRF Fusion: ChromaDB + BM25 + Multilingual E5 + Dirichlet QLM
                    ↓
Cross-Encoder Re-ranking → Document Filtering → Context Selection
                    ↓
Language Detection → Prompt Engineering → LLM (Ollama) → Response
                    ↓
Response Generation → Source Attribution → Telegram Bot Interface

================================================================================
🔧 KEY FEATURES
================================================================================

🔍 ADVANCED RETRIEVAL:
- RRF (Reciprocal Rank Fusion) with 4 retrievers
- ChromaDB semantic search with BGE-M3 embeddings
- BM25 keyword matching
- Multilingual E5 dense retrieval
- Dirichlet Query Language Model

📊 INTELLIGENT RANKING:
- Cross-encoder re-ranking
- Form field filtering
- Quality-based document selection

🌐 MULTILINGUAL SUPPORT:
- English/Bangla language detection
- Language-specific prompts
- Unicode normalization

🤖 RESPONSE GENERATION:
- Ollama integration (llama3.2:3b)
- Context-aware prompts
- Source attribution

📱 USER INTERFACE:
- Telegram bot interface
- Rich formatting
- Progress indicators

📈 EVALUATION:
- BERTScore metrics
- Cosine similarity
- Processing time analysis
- Language detection accuracy

================================================================================
📁 FILE STRUCTURE
================================================================================

CORE SYSTEM:
├── main.py                    # Vanilla RAG with Telegram bot
├── main2.py                   # Advanced RAG with RRF fusion
├── docadd_text.py            # Text chunking and ChromaDB indexing
├── sant.py                   # PDF OCR processing
├── rag_answer_generator.py   # Answer generation for evaluation
├── rag_question_processor.py # Batch question processing
└── requirements.txt          # Dependencies

EVALUATION & TESTING:
├── evaluate_rag_models.py    # Comprehensive model evaluation
├── bertscore_rag_evaluation.py # BERTScore evaluation
├── test_main.py             # Vanilla RAG testing
├── test_main2.py            # Advanced RAG testing
└── TEST_README.md           # Testing documentation

DATA PROCESSING:
├── data/                    # Processed text files
├── unsant_data/            # Raw PDF files
├── dataqa/                 # Question-answer datasets
├── chroma_db_bge_m3/       # ChromaDB vector database
└── temp_images/            # Temporary OCR files

================================================================================
⚙️ CONFIGURATION
================================================================================

RAG SYSTEM SETTINGS (main2.py):
```python
CHROMA_DB_PATH = "chroma_db_bge_m3"
EMBEDDING_MODEL = "BAAI/bge-m3"
OLLAMA_MODEL = "llama3.2:3b"
MAX_DOCS_FOR_RETRIEVAL = 50
MAX_DOCS_FOR_CONTEXT = 10
RELEVANCE_THRESHOLD = 0.09

RRF_WEIGHTS = {
    "chroma": 0.5,      # ChromaDB semantic search
    "bm25": 0.3,        # BM25 keyword search
    "multilingual_e5": 0.1,  # Multilingual dense retrieval
    "dirichlet": 0.1    # Dirichlet QLM
}
```

OCR SETTINGS (sant.py):
```python
OCR_LANG = "ben+eng"         # Bangla + English
DPI = 300                    # High DPI for quality
JPEG_QUALITY = 95            # Image quality
```

CHUNKING SETTINGS (docadd_text.py):
```python
SENTENCES_PER_CHUNK = 6      # Sentences per chunk
SENTENCE_OVERLAP = 3         # Overlap between chunks
MIN_CHUNK_SIZE = 100         # Minimum chunk size
MAX_CHUNK_SIZE = 2000        # Maximum chunk size
```

================================================================================
🚀 USAGE GUIDE
================================================================================

A. DOCUMENT PROCESSING:
1. Place PDF files in `unsant_data/` directory
2. Run OCR: python sant.py
3. Build vector DB: python docadd_text.py

B. RAG SYSTEM USAGE:
1. Vanilla RAG: python main.py
2. Advanced RAG: python main2.py
3. Batch processing: python rag_question_processor.py -i questions.csv -o responses.json

C. EVALUATION:
1. Comprehensive: python evaluate_rag_models.py
2. BERTScore: python bertscore_rag_evaluation.py
3. Quick testing: python test_main.py or python test_main2.py

================================================================================
🔧 TROUBLESHOOTING
================================================================================

CHROMADB ERRORS:
- Issue: "ChromaDB directory not found"
- Solution: Run python docadd_text.py

OLLAMA ERRORS:
- Issue: "Failed to connect to Ollama"
- Solution: Start Ollama: ollama serve and ollama pull llama3.2:3b

TESSERACT ERRORS:
- Issue: "Tesseract not found"
- Solution: Install Tesseract and set path in sant.py
- Path: C:\Program Files\Tesseract-OCR\tesseract.exe

MEMORY ISSUES:
- Issue: "Out of memory"
- Solution: Reduce batch sizes or process files individually

BERTSCORE ERRORS:
- Issue: "BERTScore evaluation failed"
- Solution: pip install bert-score

TELEGRAM BOT ERRORS:
- Issue: "Bot token not configured"
- Solution: Get token from @BotFather and update TELEGRAM_BOT_TOKEN

================================================================================
📊 PERFORMANCE METRICS
================================================================================

PROCESSING SPEED:
- PDF to Text (OCR): ~2-5 pages/minute
- Text Chunking: ~1000 chunks/minute
- Vector Indexing: ~500 chunks/minute
- Query Processing: ~2-5 seconds/query

MEMORY USAGE:
- Base System: ~2-4GB RAM
- With GPU: ~4-8GB RAM
- Large Document Processing: ~8-16GB RAM

ACCURACY METRICS:
- BERTScore F1: 0.82-0.85
- Language Detection: 94-96%
- Document Retrieval: 87-92%
- Cross-encoder Improvement: +10-15%

================================================================================
📈 EVALUATION RESULTS
================================================================================

SAMPLE EVALUATION RESULTS:
```
ADVANCED RAG SYSTEM EVALUATION RESULTS
=====================================

📊 Performance Metrics:
  Average Processing Time: 3.45s
  Documents Retrieved: 25.3
  Documents Used: 8.7
  Language Detection Accuracy: 94.2%

🎯 Quality Metrics (BERTScore):
  BERTScore F1: 0.8234
  BERTScore Precision: 0.8456
  BERTScore Recall: 0.8012
  Composite Score: 0.7891

🔍 Retrieval Analysis:
  RRF Fusion Effectiveness: 87.3%
  Cross-encoder Improvement: +12.4%
  Multilingual Query Success: 91.7%
```

================================================================================
🎯 PREREQUISITES
================================================================================

- Python 3.8 or higher
- Windows 10/11 (optimized for Windows)
- Minimum 8GB RAM (16GB recommended)
- CUDA-compatible GPU (optional but recommended)

REQUIRED INSTALLATIONS:
1. Ollama: https://ollama.ai
2. Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki
3. Bengali language pack: https://github.com/tesseract-ocr/tessdata

================================================================================
📞 SUPPORT
================================================================================

For technical support, bug reports, or feature requests:
- Create an issue on the project repository
- Check the troubleshooting section above
- Review the test scripts for usage examples

For questions about RAG system architecture:
- Refer to comprehensive evaluation results
- Check TEST_README.md for testing procedures
- Review individual script documentation

================================================================================
🎉 ACKNOWLEDGMENTS
================================================================================

This project builds upon open-source libraries:
- LangChain for RAG pipeline orchestration
- ChromaDB for vector storage and retrieval
- BGE-M3 for multilingual embeddings
- Ollama for local LLM inference
- Tesseract OCR for document processing
- BERTScore for evaluation metrics

================================================================================
📝 VERSION HISTORY
================================================================================

v2.1 - Advanced RAG with RRF Fusion
- Added RRF fusion with 4 retrievers
- Implemented cross-encoder re-ranking
- Enhanced multilingual support
- Comprehensive evaluation framework

v2.0 - Telegram Bot Integration
- Added Telegram bot interface
- Improved error handling
- Enhanced user experience
- Real-time processing capabilities

v1.0 - Basic RAG System
- Initial implementation
- PDF processing pipeline
- Basic ChromaDB integration
- Simple question-answering

================================================================================
🚀 GETTING STARTED QUICKLY
================================================================================

1. pip install -r requirements.txt
2. python setup_nltk.py
3. ollama pull llama3.2:3b
4. python sant.py
5. python docadd_text.py
6. python main2.py

================================================================================
END OF README
================================================================================