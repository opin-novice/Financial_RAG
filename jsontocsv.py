#!/usr/bin/env python3
"""
Script to extract questions and responses from qaR.json to a CSV file.
Default: python jsontocsv.py (creates questions_responses.csv)
Custom: python jsontocsv.py input.json output.csv
"""

import json
import pandas as pd
import sys
from pathlib import Path
import re


def clean_response_text(response_text):
    """
    Clean malformed response text by removing extra newlines and formatting issues.
    
    Args:
        response_text (str): Raw response text
        
    Returns:
        str: Cleaned response text
    """
    if not response_text:
        return ""
    
    # Remove excessive newlines and replace with single spaces
    cleaned = re.sub(r'\n+', ' ', response_text.strip())
    
    # Remove extra whitespaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Replace bullet points with commas for better readability
    cleaned = re.sub(r'\s*[•·\-\*]\s*', ', ', cleaned)
    
    # Clean up leading comma if it exists
    cleaned = re.sub(r'^,\s*', '', cleaned)
    
    # Clean up multiple commas
    cleaned = re.sub(r',\s*,', ',', cleaned)
    
    return cleaned.strip()


def extract_questions_responses(json_file_path, csv_file_path):
    """
    Extract questions and responses from qaR.json and save to CSV file using pandas DataFrame.
    
    Args:
        json_file_path (str): Path to the qaR.json file
        csv_file_path (str): Path to the output CSV file
    """
    try:
        # Read the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        
        # Check if the expected structure exists
        if 'questions_and_responses' not in data:
            print("Error: 'questions_and_responses' key not found in JSON file")
            return False
        
        questions_responses = data['questions_and_responses']
        
        # Prepare data for DataFrame
        data_list = []
        for item in questions_responses:
            question = item.get('question', '').strip()
            response = clean_response_text(item.get('response', ''))
            
            # Only add entries that have both question and response
            if question and response:
                data_list.append({
                    'question': question,
                    'response': response
                })
        
        # Create DataFrame
        df = pd.DataFrame(data_list)
        
        # Save to CSV
        df.to_csv(csv_file_path, index=False, encoding='utf-8')
        
        print(f"Successfully extracted {len(data_list)} questions and responses to {csv_file_path}")
        print(f"DataFrame shape: {df.shape}")
        return True
        
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found")
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {json_file_path}: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Main function to handle command line arguments and execute extraction."""
    
    # Default file paths
    json_file = "qaR.json"
    csv_file = "questions_responses.csv"
    
    # Check if custom file paths are provided as command line arguments
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    if len(sys.argv) > 2:
        csv_file = sys.argv[2]
    
    # Check if JSON file exists
    if not Path(json_file).exists():
        print(f"Error: JSON file '{json_file}' does not exist")
        print("Usage: python jsontocsv.py [json_file] [csv_file]")
        sys.exit(1)
    
    # Extract questions and responses
    success = extract_questions_responses(json_file, csv_file)
    
    if success:
        print(f"Extraction completed successfully!")
        print(f"Input file: {json_file}")
        print(f"Output file: {csv_file}")
    else:
        print("Extraction failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
