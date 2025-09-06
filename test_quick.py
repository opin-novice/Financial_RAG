# test_quick.py
from main2 import CoreRAGSystem

# Initialize the system
print("🚀 Initializing Advanced RAG System...")
rag = CoreRAGSystem()

# Add this debug code here:
print(" DEBUGGING METADATA MAPPING:")
print(f"Total metadata entries: {len(rag.retriever.metadata_mapping)}")
if rag.retriever.metadata_mapping:
    first_key = list(rag.retriever.metadata_mapping.keys())[0]
    first_metadata = rag.retriever.metadata_mapping[first_key]
    print(f"First metadata entry: {first_metadata}")
    print(f"Available keys: {list(first_metadata.keys())}")
else:
    print("No metadata mapping found!")

# Test with a few questions
test_questions = [
    "What is the short title of the rules made under section 185 of the Income Tax Ordinance, 1984?",
    "What are the types of investment qualified for the tax rebate?",
    "What is VAT Assessment?"
]

for i, question in enumerate(test_questions):
    print(f"\n{'='*60}")
    print(f"Question {i+1}: {question}")
    print('='*60)
    
    result = rag.process_query(question)

    # Add this debug line:
    print(f"DEBUG - First source metadata: {result.get('sources', [{}])[0]}")
    print(f"DEBUG - All metadata keys: {list(result.get('sources', [{}])[0].keys())}")
    
    print(f"Response: {result['response'][:200]}...")
    print(f"Documents found: {result.get('documents_found', 0)}")
    print(f"Documents used: {result.get('documents_used', 0)}")
    print(f"Sources:")
    for j, source in enumerate(result.get('sources', [])[:3]):  # Show first 3 sources
        print(f"  {j+1}. {source['file']} - Page: {source['page']} - Score: {source['score']:.3f}")
