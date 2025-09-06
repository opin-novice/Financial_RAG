#!/usr/bin/env python3
"""
📊 CSV to JSON Converter (Question-Answer Format)
=================================================

This script converts CSV files containing questions and answers to a clean JSON format
using pandas DataFrame with flexible file selection options.

Output Format:
[
    {
        "question": "What is the corporate tax rate?", 
        "answer": "The corporate tax rate is 27.5%."
    },
    ...
]

Features:
- Flexible input: Select any CSV file with question-answer pairs
- Customizable output: Name your JSON output file as desired
- Command-line interface with multiple usage modes
- Interactive mode for easy file selection
- Automatic CSV column detection (question/answer columns)
- Automatic CSV encoding detection
- Clean JSON output optimized for Q&A datasets
- Progress tracking and comprehensive error handling

Usage Modes:
1. Default: python csvtojson.py
2. Command-line: python csvtojson.py -i input.csv -o output.json
3. Interactive: python csvtojson.py --interactive
4. List CSV files: python csvtojson.py --list-csv

Author: CSV to JSON Converter
Version: 2.0
"""

import os
import sys
import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="📊 CSV to JSON Converter - Convert CSV files to JSON format using pandas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python csvtojson.py
  python csvtojson.py -i data.csv -o data.json
  python csvtojson.py --input questions.csv --output questions.json
  python csvtojson.py --interactive
  python csvtojson.py -i data.csv -o data.json --indent 0
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        help='Input CSV file to convert'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output JSON file name'
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
    
    parser.add_argument(
        '--indent',
        type=int,
        default=2,
        help='JSON indentation spaces (default: 2, use 0 for compact)'
    )
    
    parser.add_argument(
        '--encoding',
        type=str,
        default='auto',
        help='CSV file encoding (default: auto-detect)'
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
        file_size = csv_file.stat().st_size
        size_str = format_file_size(file_size)
        print(f"   {i}. {csv_file.name} ({size_str})")
    
    return csv_files


def format_file_size(size_bytes):
    """Format file size in human-readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def detect_csv_encoding(file_path):
    """Detect CSV file encoding"""
    try:
        import chardet
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB
            result = chardet.detect(raw_data)
            return result['encoding']
    except ImportError:
        # If chardet is not available, try common encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1000)  # Try to read first 1KB
                return encoding
            except UnicodeDecodeError:
                continue
        return 'utf-8'  # Default fallback


def interactive_file_selection():
    """Interactive mode for file selection"""
    print("📊 CSV to JSON Converter - Interactive Mode")
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
        default_output = Path(input_csv).stem + ".json"
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


def load_csv_data(file_path: str, encoding: str = 'auto') -> pd.DataFrame:
    """Load CSV data using pandas with encoding detection"""
    try:
        # Detect encoding if auto
        if encoding == 'auto':
            encoding = detect_csv_encoding(file_path)
            print(f"🔍 Detected encoding: {encoding}")
        
        print(f"📖 Loading CSV file: {file_path}")
        
        # Load CSV with pandas
        df = pd.read_csv(file_path, encoding=encoding)
        
        print(f"✅ CSV loaded successfully!")
        print(f"   - Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"   - Columns: {list(df.columns)}")
        
        # Show data types
        print(f"   - Data types:")
        for col, dtype in df.dtypes.items():
            print(f"     • {col}: {dtype}")
        
        return df
        
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {file_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file: {e}")
    except UnicodeDecodeError as e:
        raise ValueError(f"Encoding error reading CSV file: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error loading CSV: {e}")


def detect_qa_columns(df: pd.DataFrame) -> tuple:
    """Detect question and answer columns in the DataFrame"""
    columns = [col.lower() for col in df.columns]
    
    # Common question column names
    question_names = ['question', 'questions', 'query', 'queries', 'q', 'prompt']
    # Common answer column names  
    answer_names = ['answer', 'answers', 'response', 'responses', 'reply', 'solution', 'a', 'reference_answer']
    
    question_col = None
    answer_col = None
    
    # Find question column
    for q_name in question_names:
        for i, col in enumerate(columns):
            if q_name in col:
                question_col = df.columns[i]
                break
        if question_col:
            break
    
    # Find answer column
    for a_name in answer_names:
        for i, col in enumerate(columns):
            if a_name in col:
                answer_col = df.columns[i]
                break
        if answer_col:
            break
    
    # If not found, use first two columns
    if not question_col and len(df.columns) >= 1:
        question_col = df.columns[0]
    if not answer_col and len(df.columns) >= 2:
        answer_col = df.columns[1]
    
    return question_col, answer_col


def convert_to_qa_json(df: pd.DataFrame, output_path: str, indent: int = 2) -> bool:
    """Convert DataFrame to question-answer JSON format"""
    try:
        print(f"🔄 Converting to question-answer JSON format...")
        
        # Detect question and answer columns
        question_col, answer_col = detect_qa_columns(df)
        
        if not question_col:
            raise ValueError("Could not detect question column in CSV")
        if not answer_col:
            raise ValueError("Could not detect answer column in CSV")
        
        print(f"   - Question column: '{question_col}'")
        print(f"   - Answer column: '{answer_col}'")
        
        # Create question-answer pairs
        qa_pairs = []
        for _, row in df.iterrows():
            question = str(row[question_col]).strip()
            answer = str(row[answer_col]).strip()
            
            # Skip rows with empty questions or answers
            if question and answer and question != 'nan' and answer != 'nan':
                qa_pair = {
                    "question": question,
                    "answer": answer
                }
                qa_pairs.append(qa_pair)
        
        if not qa_pairs:
            raise ValueError("No valid question-answer pairs found in the CSV")
        
        print(f"   - Created {len(qa_pairs)} question-answer pairs")
        
        # Write JSON file
        print(f"💾 Saving JSON file: {output_path}")
        
        indent_value = indent if indent > 0 else None
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=indent_value)
        
        # Get file size
        file_size = Path(output_path).stat().st_size
        size_str = format_file_size(file_size)
        
        print(f"✅ JSON file saved successfully!")
        print(f"   - Output file: {output_path}")
        print(f"   - File size: {size_str}")
        print(f"   - Format: question-answer pairs")
        print(f"   - Total pairs: {len(qa_pairs)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting to JSON: {e}")
        return False


def preview_data(df: pd.DataFrame, num_rows: int = 3):
    """Preview the first few rows of the DataFrame"""
    print(f"\n📋 Data Preview (first {num_rows} rows):")
    print("=" * 60)
    print(df.head(num_rows).to_string(index=False))
    print("=" * 60)


def main():
    """Main function to handle CSV to JSON conversion"""
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
        encoding = 'auto'
        indent = 2
    else:
        # Command-line mode
        if not args.input:
            print("❌ Error: Input CSV file is required")
            print("Use --help for usage information or --interactive for interactive mode")
            return 1
        
        input_csv = args.input
        output_json = args.output or (Path(input_csv).stem + ".json")
        encoding = args.encoding
        indent = args.indent
    
    # Display configuration
    print("📊 CSV to JSON Converter")
    print("=" * 50)
    print(f"Input CSV: {input_csv}")
    print(f"Output JSON: {output_json}")
    print(f"Format: Question-Answer pairs")
    print(f"Encoding: {encoding}")
    print("=" * 50)
    
    try:
        # Check if input file exists
        if not Path(input_csv).exists():
            print(f"❌ Error: Input CSV file '{input_csv}' not found!")
            return 1
        
        # Confirm overwrite if output file exists
        if Path(output_json).exists():
            print(f"\n⚠️ Output file '{output_json}' already exists!")
            if args.interactive:
                confirm = input("Do you want to overwrite it? (y/N): ").strip().lower()
                if confirm not in ['y', 'yes']:
                    print("❌ Operation cancelled")
                    return 1
            else:
                print("⚠️ File will be overwritten")
        
        # Load CSV data
        df = load_csv_data(input_csv, encoding)
        
        # Preview data in interactive mode
        if args.interactive and not df.empty:
            preview = input("\n🔍 Would you like to preview the data? (y/N): ").strip().lower()
            if preview in ['y', 'yes']:
                preview_data(df)
        
        # Convert to JSON
        success = convert_to_qa_json(df, output_json, indent)
        
        if success:
            print(f"\n🎉 Conversion completed successfully!")
            return 0
        else:
            print(f"\n❌ Conversion failed!")
            return 1
        
    except KeyboardInterrupt:
        print("\n⚠️ Conversion interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
