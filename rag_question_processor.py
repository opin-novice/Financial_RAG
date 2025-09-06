#!/usr/bin/env python3
"""
🤖 RAG Question Processor
=========================

This script processes questions from any CSV file using the main2.py RAG system
and stores the questions and responses in a customizable JSON file.

Features:
- Flexible input: Select any CSV file with questions
- Customizable output: Name your JSON output file as desired
- Command-line interface with multiple usage modes
- Interactive mode for easy file selection
- Automatic CSV format detection and flexible column naming
- Handles both English and Bengali questions
- Provides detailed metadata for each response
- Progress tracking and comprehensive error handling
- File existence checks and overwrite confirmation

Usage Modes:
1. Default: python rag_question_processor.py
2. Command-line: python rag_question_processor.py -i input.csv -o output.json
3. Interactive: python rag_question_processor.py --interactive
4. List CSV files: python rag_question_processor.py --list-csv

Author: RAG Question Processor
Version: 2.0
"""

import os
import csv
import json
import logging
import time
import sys
import argparse
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

# Import RAG system from main2.py
from main2 import CoreRAGSystem

# =============================================================================
# 📊 CONFIGURATION
# =============================================================================

# File paths
INPUT_CSV_FILE = "RAG_questions.csv"
OUTPUT_JSON_FILE = "qaR.json"

# Processing settings
BATCH_SIZE = 10  # Process questions in batches for progress updates
DELAY_BETWEEN_QUESTIONS = 0.5  # Seconds to wait between questions (to avoid overload)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# 📝 QUESTION PROCESSOR
# =============================================================================

class RAGQuestionProcessor:
    """Processes questions from CSV using RAG system and saves responses to JSON"""
    
    def __init__(self, input_csv: str = INPUT_CSV_FILE, output_json: str = OUTPUT_JSON_FILE):
        """Initialize the question processor
        
        Args:
            input_csv: Path to the input CSV file with questions
            output_json: Path to the output JSON file for responses
        """
        self.input_csv = input_csv
        self.output_json = output_json
        self.rag_system = None
        
        logger.info("🚀 Initializing RAG Question Processor...")
        self._init_rag_system()
        logger.info("✅ RAG Question Processor initialized successfully")
    
    def _init_rag_system(self):
        """Initialize the RAG system from main2.py"""
        try:
            logger.info("Loading RAG system from main2.py...")
            self.rag_system = CoreRAGSystem()
            
            # Verify system is ready
            info = self.rag_system.get_system_info()
            if not info['faiss_index_loaded']:
                raise Exception("FAISS index not loaded")
            if not info['embedding_model_loaded']:
                raise Exception("Embedding model not loaded")
            if not info['llm_initialized']:
                raise Exception("LLM not initialized")
            
            logger.info("✅ RAG system initialized successfully")
            logger.info(f"   - Documents: {info['total_vectors']:,}")
            logger.info(f"   - FAISS Index: {'✅' if info['faiss_index_loaded'] else '❌'}")
            logger.info(f"   - LLM Model: {info['config']['ollama_model']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def load_questions_from_csv(self) -> List[Dict[str, Any]]:
        """Load questions from CSV file"""
        try:
            logger.info(f"📄 Loading questions from: {self.input_csv}")
            
            if not os.path.exists(self.input_csv):
                raise FileNotFoundError(f"CSV file not found: {self.input_csv}")
            
            questions = []
            with open(self.input_csv, 'r', encoding='utf-8') as csvfile:
                # Try to detect the CSV format
                sample = csvfile.read(1024)
                csvfile.seek(0)
                
                # Detect delimiter
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                
                reader = csv.DictReader(csvfile, delimiter=delimiter)
                
                # Get the question column name (flexible naming)
                fieldnames = reader.fieldnames
                question_column = None
                
                # Look for common question column names
                possible_names = ['question', 'questions', 'query', 'queries', 'q', 'Question', 'Query']
                for name in possible_names:
                    if name in fieldnames:
                        question_column = name
                        break
                
                if not question_column:
                    # Use the first column if no standard name found
                    question_column = fieldnames[0]
                    logger.warning(f"No standard question column found. Using '{question_column}'")
                
                logger.info(f"📝 Using column '{question_column}' for questions")
                
                # Read questions
                for i, row in enumerate(reader):
                    question_text = row.get(question_column, '').strip()
                    if question_text:
                        question_data = {
                            'id': i + 1,
                            'question': question_text,
                            'original_row': row  # Keep original row data
                        }
                        questions.append(question_data)
            
            logger.info(f"✅ Loaded {len(questions)} questions from CSV")
            return questions
            
        except Exception as e:
            logger.error(f"Failed to load questions from CSV: {e}")
            raise
    
    def process_single_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single question through the RAG system"""
        question_id = question_data['id']
        question_text = question_data['question']
        
        logger.info(f"🔍 Processing question {question_id}: {question_text[:50]}...")
        
        try:
            # Process question through RAG system
            start_time = time.time()
            result = self.rag_system.process_query_sync(question_text)
            processing_time = time.time() - start_time
            
            # Create response data
            response_data = {
                'id': question_id,
                'question': question_text,
                'response': result.get('response', ''),
                'metadata': {
                    'detected_language': result.get('detected_language', 'unknown'),
                    'language_confidence': result.get('language_confidence', 0.0),
                    'retrieval_method': result.get('retrieval_method', 'unknown'),
                    'documents_found': result.get('documents_found', 0),
                    'documents_used': result.get('documents_used', 0),
                    'cross_encoder_used': result.get('cross_encoder_used', False),
                    'processing_time': round(processing_time, 2)
                },
                'sources': result.get('sources', []),
                'contexts': result.get('contexts', []),
                'timestamp': datetime.now().isoformat(),
                'original_row_data': question_data.get('original_row', {})
            }
            
            logger.info(f"✅ Question {question_id} processed successfully")
            logger.info(f"   - Language: {response_data['metadata']['detected_language']}")
            logger.info(f"   - Documents used: {response_data['metadata']['documents_used']}")
            logger.info(f"   - Processing time: {response_data['metadata']['processing_time']}s")
            
            return response_data
            
        except Exception as e:
            logger.error(f"❌ Failed to process question {question_id}: {e}")
            
            # Return error response
            error_response = {
                'id': question_id,
                'question': question_text,
                'response': f"Error processing question: {str(e)}",
                'metadata': {
                    'error': str(e),
                    'processing_time': 0.0,
                    'status': 'failed'
                },
                'sources': [],
                'contexts': [],
                'timestamp': datetime.now().isoformat(),
                'original_row_data': question_data.get('original_row', {})
            }
            return error_response
    
    def process_all_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process all questions through the RAG system"""
        logger.info(f"🔄 Processing {len(questions)} questions through RAG system...")
        
        responses = []
        successful_count = 0
        failed_count = 0
        
        for i, question_data in enumerate(questions):
            try:
                # Process the question
                response = self.process_single_question(question_data)
                responses.append(response)
                
                # Check if successful
                if 'error' not in response['metadata']:
                    successful_count += 1
                else:
                    failed_count += 1
                
                # Progress update
                if (i + 1) % BATCH_SIZE == 0 or (i + 1) == len(questions):
                    logger.info(f"📊 Progress: {i + 1}/{len(questions)} questions processed")
                    logger.info(f"   - Successful: {successful_count}, Failed: {failed_count}")
                
                # Small delay to avoid overwhelming the system
                if DELAY_BETWEEN_QUESTIONS > 0 and i < len(questions) - 1:
                    time.sleep(DELAY_BETWEEN_QUESTIONS)
                    
            except Exception as e:
                logger.error(f"❌ Critical error processing question {question_data['id']}: {e}")
                failed_count += 1
                continue
        
        logger.info(f"✅ Question processing completed!")
        logger.info(f"   - Total processed: {len(responses)}")
        logger.info(f"   - Successful: {successful_count}")
        logger.info(f"   - Failed: {failed_count}")
        
        return responses
    
    def save_responses_to_json(self, responses: List[Dict[str, Any]]):
        """Save responses to JSON file"""
        try:
            logger.info(f"💾 Saving responses to: {self.output_json}")
            
            # Create output data structure
            output_data = {
                'processing_info': {
                    'timestamp': datetime.now().isoformat(),
                    'input_file': self.input_csv,
                    'output_file': self.output_json,
                    'total_questions': len(responses),
                    'successful_responses': len([r for r in responses if 'error' not in r['metadata']]),
                    'failed_responses': len([r for r in responses if 'error' in r['metadata']]),
                    'rag_system_info': self.rag_system.get_system_info() if self.rag_system else {}
                },
                'questions_and_responses': responses
            }
            
            # Save to JSON file
            with open(self.output_json, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Responses saved successfully to: {self.output_json}")
            
        except Exception as e:
            logger.error(f"Failed to save responses to JSON: {e}")
            raise
    
    def run_processing(self) -> str:
        """Run the complete question processing pipeline"""
        try:
            logger.info("🚀 Starting RAG Question Processing...")
            
            # Step 1: Load questions from CSV
            questions = self.load_questions_from_csv()
            
            if not questions:
                raise Exception("No questions found in CSV file")
            
            # Step 2: Process all questions
            responses = self.process_all_questions(questions)
            
            # Step 3: Save responses to JSON
            self.save_responses_to_json(responses)
            
            logger.info("✅ RAG Question Processing completed successfully!")
            return self.output_json
            
        except Exception as e:
            logger.error(f"❌ RAG Question Processing failed: {e}")
            raise
    
    def print_summary(self):
        """Print processing summary"""
        if not os.path.exists(self.output_json):
            print("❌ Output file not found. Processing may have failed.")
            return
        
        try:
            with open(self.output_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            info = data.get('processing_info', {})
            responses = data.get('questions_and_responses', [])
            
            print("\n" + "="*60)
            print("📊 RAG QUESTION PROCESSING SUMMARY")
            print("="*60)
            
            print(f"\n📄 Files:")
            print(f"   Input CSV: {info.get('input_file', 'Unknown')}")
            print(f"   Output JSON: {info.get('output_file', 'Unknown')}")
            
            print(f"\n📈 Processing Statistics:")
            print(f"   Total Questions: {info.get('total_questions', 0)}")
            print(f"   Successful Responses: {info.get('successful_responses', 0)}")
            print(f"   Failed Responses: {info.get('failed_responses', 0)}")
            
            if responses:
                languages = {}
                avg_processing_time = 0
                valid_times = 0
                
                for response in responses:
                    # Count languages
                    lang = response.get('metadata', {}).get('detected_language', 'unknown')
                    languages[lang] = languages.get(lang, 0) + 1
                    
                    # Calculate average processing time
                    proc_time = response.get('metadata', {}).get('processing_time', 0)
                    if proc_time > 0:
                        avg_processing_time += proc_time
                        valid_times += 1
                
                print(f"\n🌐 Language Distribution:")
                for lang, count in languages.items():
                    percentage = (count / len(responses)) * 100
                    print(f"   {lang.title()}: {count} questions ({percentage:.1f}%)")
                
                if valid_times > 0:
                    avg_time = avg_processing_time / valid_times
                    print(f"\n⏱️ Average Processing Time: {avg_time:.2f} seconds")
            
            print("\n" + "="*60)
            
        except Exception as e:
            print(f"❌ Error reading summary: {e}")

# =============================================================================
# 🚀 MAIN EXECUTION
# =============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="🤖 RAG Question Processor - Process questions from CSV using RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rag_question_processor.py
  python rag_question_processor.py -i questions.csv -o responses.json
  python rag_question_processor.py --input my_questions.csv --output my_responses.json
  python rag_question_processor.py --interactive
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        default=INPUT_CSV_FILE,
        help=f'Input CSV file with questions (default: {INPUT_CSV_FILE})'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=OUTPUT_JSON_FILE,
        help=f'Output JSON file for responses (default: {OUTPUT_JSON_FILE})'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode to select files'
    )
    
    parser.add_argument(
        '--list-csv',
        action='store_true',
        help='List available CSV files in the current directory'
    )
    
    return parser.parse_args()


def list_csv_files():
    """List available CSV files in the current directory"""
    csv_files = list(Path('.').glob('*.csv'))
    
    if not csv_files:
        print("❌ No CSV files found in the current directory")
        return []
    
    print("📄 Available CSV files:")
    for i, csv_file in enumerate(csv_files, 1):
        print(f"   {i}. {csv_file.name}")
    
    return csv_files


def interactive_file_selection():
    """Interactive mode for file selection"""
    print("🤖 RAG Question Processor - Interactive Mode")
    print("=" * 50)
    
    # Select input CSV file
    csv_files = list_csv_files()
    if not csv_files:
        return None, None
    
    while True:
        try:
            print(f"\n📥 Select input CSV file (1-{len(csv_files)}) or enter custom path:")
            choice = input("Enter your choice: ").strip()
            
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(csv_files):
                    input_csv = str(csv_files[index])
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(csv_files)}")
            else:
                # Custom path
                if Path(choice).exists() and choice.endswith('.csv'):
                    input_csv = choice
                    break
                else:
                    print("❌ File not found or not a CSV file. Please try again.")
        except ValueError:
            print("❌ Invalid input. Please try again.")
    
    # Select output JSON file
    while True:
        default_output = Path(input_csv).stem + "_responses.json"
        print(f"\n📤 Enter output JSON file name (default: {default_output}):")
        output_choice = input("Output file: ").strip()
        
        if not output_choice:
            output_json = default_output
            break
        else:
            if not output_choice.endswith('.json'):
                output_choice += '.json'
            output_json = output_choice
            break
    
    return input_csv, output_json


def main():
    """Main function to run the RAG question processing"""
    args = parse_arguments()
    
    # Handle list CSV files option
    if args.list_csv:
        list_csv_files()
        return 0
    
    # Handle interactive mode
    if args.interactive:
        input_csv, output_json = interactive_file_selection()
        if not input_csv or not output_json:
            print("❌ File selection cancelled")
            return 1
    else:
        input_csv = args.input
        output_json = args.output
    
    # Display configuration
    print("🤖 RAG Question Processor")
    print("=" * 50)
    print(f"Input CSV: {input_csv}")
    print(f"Output JSON: {output_json}")
    print("=" * 50)
    
    try:
        # Check if input file exists
        if not os.path.exists(input_csv):
            print(f"❌ Error: Input CSV file '{input_csv}' not found!")
            print("\nPlease create a CSV file with questions. Example format:")
            print("question")
            print("What documents are required for a car loan?")
            print("How to open a savings account?")
            print("What is the interest rate for personal loans?")
            return 1
        
        # Check if RAG system requirements exist
        if not os.path.exists("faiss_index"):
            print("❌ Error: FAISS index directory 'faiss_index' not found!")
            print("Please make sure the RAG system is properly set up.")
            return 1
        
        # Confirm overwrite if output file exists
        if os.path.exists(output_json):
            print(f"\n⚠️ Output file '{output_json}' already exists!")
            if args.interactive:
                confirm = input("Do you want to overwrite it? (y/N): ").strip().lower()
                if confirm not in ['y', 'yes']:
                    print("❌ Operation cancelled")
                    return 1
            else:
                print("⚠️ File will be overwritten")
        
        # Initialize and run processor
        processor = RAGQuestionProcessor(input_csv, output_json)
        output_file = processor.run_processing()
        
        # Print summary
        processor.print_summary()
        
        print(f"\n🎉 Processing completed successfully!")
        print(f"📄 Results saved to: {output_file}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        logger.error(f"Processing failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
