#!/usr/bin/env python3
"""
Advanced PDF Text Extraction using Image-based OCR
Uses pypdfium2 for PDF to image conversion + Tesseract OCR with preprocessing
Specifically optimized for Bengali text extraction
"""
import os
import re
import cv2
import numpy as np
import pytesseract
import pypdfium2 as pdfium
from PIL import Image, ImageEnhance, ImageFilter
import unicodedata
from datetime import datetime

# ✅ Explicitly set Tesseract path (important for Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- CONFIGURATION ---
INPUT_FOLDER = "unsant_data"
OUTPUT_FOLDER = "data"
TEMP_FOLDER = "temp_images"  # For storing temporary JPEG images
OCR_LANG = "ben+eng"         # Bangla + English OCR
OUTPUT_FORMAT = "txt"        # Use text files for better Unicode support
DPI = 300                    # High DPI for better OCR quality
JPEG_QUALITY = 95            # High quality JPEG for processing

# Create temp folder if it doesn't exist
os.makedirs(TEMP_FOLDER, exist_ok=True)

def detect_language(text: str) -> str:
    """
    Detect if text is in Bangla, English, or both.
    
    Args:
        text: Text to analyze
        
    Returns:
        Language code: 'bn', 'en', or 'bn+en'
    """
    if not text:
        return 'unknown'
    
    # Check for Bangla characters (Unicode range U+0980–U+09FF)
    has_bangla = bool(re.search(r'[\u0980-\u09FF]', text))
    
    # Check for English characters
    has_english = bool(re.search(r'[a-zA-Z]', text))
    
    if has_bangla and has_english:
        return 'bn+en'
    elif has_bangla:
        return 'bn'
    elif has_english:
        return 'en'
    else:
        return 'unknown'

def pdf_to_images(pdf_path, temp_folder=TEMP_FOLDER):
    """
    Convert each page of PDF to high-quality JPEG images using pypdfium2.
    
    Args:
        pdf_path: Path to the PDF file
        temp_folder: Folder to store temporary images
        
    Returns:
        List of image file paths
    """
    try:
        # Open PDF with pypdfium2
        pdf = pdfium.PdfDocument(pdf_path)
        image_paths = []
        
        print(f"[INFO] Converting {len(pdf)} pages to images...")
        
        for page_num in range(len(pdf)):
            # Get page and render to bitmap
            page = pdf.get_page(page_num)
            bitmap = page.render(
                scale=DPI/72,  # Convert DPI to scale factor
                rotation=0,
                crop=(0, 0, 0, 0)
            )
            
            # Convert to PIL Image
            pil_image = bitmap.to_pil()
            
            # Save as high-quality JPEG
            image_path = os.path.join(temp_folder, f"page_{page_num + 1}.jpg")
            pil_image.save(image_path, "JPEG", quality=JPEG_QUALITY, optimize=True)
            image_paths.append(image_path)
            
            print(f"[INFO] ✅ Page {page_num + 1} saved as {image_path}")
        
        pdf.close()
        return image_paths
        
    except Exception as e:
        print(f"[ERROR] Failed to convert PDF to images: {e}")
        return []

def advanced_preprocess_image(image_path):
    """
    Apply advanced preprocessing techniques for better Bengali OCR.
    
    Steps:
    1. Median filtering for noise reduction
    2. Contrast enhancement
    3. Black & white conversion with adaptive thresholding
    4. Morphological operations for text clarity
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image as numpy array
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"[INFO] 🔧 Preprocessing {os.path.basename(image_path)}...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Median filtering for noise reduction
        # This removes salt-and-pepper noise while preserving edges
        denoised = cv2.medianBlur(gray, 3)
        
        # Step 2: Contrast enhancement using CLAHE
        # Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Step 3: Additional contrast enhancement
        # Normalize to full range
        normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        
        # Step 4: Black & white conversion with adaptive thresholding
        # This works better than simple thresholding for varied lighting
        binary = cv2.adaptiveThreshold(
            normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Step 5: Morphological operations for text clarity
        # Remove small noise and connect broken characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Step 6: Final noise removal
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        final = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel2)
        
        print(f"[INFO] ✅ Preprocessing complete for {os.path.basename(image_path)}")
        return final
        
    except Exception as e:
        print(f"[ERROR] Preprocessing failed for {image_path}: {e}")
        return None

def extract_text_from_image(image_array):
    """
    Extract text from preprocessed image using Tesseract OCR.
    
    Args:
        image_array: Preprocessed image as numpy array
        
    Returns:
        Extracted text string
    """
    try:
        # Configure Tesseract for Bengali + English
        custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        
        # Extract text
        text = pytesseract.image_to_string(
            image_array, 
            lang=OCR_LANG, 
            config=custom_config
        )
        
        return text
        
    except Exception as e:
        print(f"[ERROR] OCR extraction failed: {e}")
        return ""

def clean_extracted_text(text):
    """
    Clean and normalize extracted text.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Unicode normalization
    text = unicodedata.normalize('NFKC', text)
    
    # Remove excessive whitespace
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
    text = re.sub(r' *\n *', '\n', text)  # Remove spaces around newlines
    text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double
    
    # Remove common OCR artifacts
    text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE)  # Page numbers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # Standalone numbers
    
    # Clean up dots and underscores
    text = re.sub(r'\.{5,}', '', text)  # 5+ dots
    text = re.sub(r'_{5,}', '', text)   # 5+ underscores
    
    return text.strip()

def process_pdf_to_text(pdf_path):
    """
    Main processing function: PDF -> Images -> OCR -> Text
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted and cleaned text
    """
    print(f"[INFO] 🚀 Processing {os.path.basename(pdf_path)} with image-based OCR...")
    
    # Step 1: Convert PDF pages to images
    image_paths = pdf_to_images(pdf_path)
    if not image_paths:
        print(f"[ERROR] Failed to convert PDF to images: {pdf_path}")
        return ""
    
    # Step 2: Process each image with OCR
    all_text = ""
    successful_pages = 0
    
    for i, image_path in enumerate(image_paths):
        print(f"[INFO] 📄 Processing page {i + 1}/{len(image_paths)}...")
        
        # Preprocess image
        preprocessed = advanced_preprocess_image(image_path)
        if preprocessed is None:
            print(f"[WARNING] Skipping page {i + 1} due to preprocessing failure")
            continue
        
        # Extract text with OCR
        page_text = extract_text_from_image(preprocessed)
        if page_text.strip():
            cleaned_text = clean_extracted_text(page_text)
            if cleaned_text and len(cleaned_text.strip()) > 10:  # Minimum length check
                all_text += cleaned_text + "\n\n"
                successful_pages += 1
                print(f"[INFO] ✅ Page {i + 1}: Extracted {len(cleaned_text)} characters")
            else:
                print(f"[INFO] ⚠️ Page {i + 1}: Minimal text extracted")
        else:
            print(f"[INFO] ⚠️ Page {i + 1}: No text extracted")
    
    # Step 3: Clean up temporary images
    for image_path in image_paths:
        try:
            os.remove(image_path)
        except:
            pass
    
    print(f"[INFO] 📊 Successfully processed {successful_pages}/{len(image_paths)} pages")
    return all_text.strip()

def create_output_file(text, pdf_path, output_folder=OUTPUT_FOLDER):
    """
    Save extracted text to output file with metadata.
    
    Args:
        text: Extracted text
        pdf_path: Original PDF path
        output_folder: Output directory
        
    Returns:
        Path to created file or None if failed
    """
    try:
        # Create output filename
        pdf_name = os.path.basename(pdf_path)
        txt_name = pdf_name.replace('.pdf', '.txt')
        output_path = os.path.join(output_folder, txt_name)
        
        # Create metadata header
        header = f"""=== Advanced OCR Text Extraction Result ===
Original File: {pdf_name}
Extraction Method: Image-based OCR (pypdfium2 + Tesseract)
Language: {detect_language(text)}
Extracted Characters: {len(text)}
Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{"=" * 60}

"""
        
        # Save file with UTF-8 encoding
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(header + text)
        
        print(f"[INFO] ✅ Output saved: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"[ERROR] Failed to save output file: {e}")
        return None

def sanitize_pdfs_advanced(input_folder=INPUT_FOLDER, output_folder=OUTPUT_FOLDER):
    """
    Main function to process all PDFs in input folder using advanced image-based OCR.
    
    Args:
        input_folder: Folder containing PDF files
        output_folder: Folder to save extracted text files
    """
    print(f"[INFO] 🚀 Starting advanced image-based PDF text extraction...")
    print(f"[INFO] Input folder: {input_folder}")
    print(f"[INFO] Output folder: {output_folder}")
    print(f"[INFO] OCR Language: {OCR_LANG}")
    print(f"[INFO] Image DPI: {DPI}")
    
    # Create output folder if needed
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all PDF files
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"[WARNING] No PDF files found in {input_folder}")
        return
    
    print(f"[INFO] Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    successful = 0
    for i, filename in enumerate(pdf_files):
        print(f"\n[INFO] 📁 Processing file {i + 1}/{len(pdf_files)}: {filename}")
        
        input_path = os.path.join(input_folder, filename)
        
        # Extract text using image-based OCR
        extracted_text = process_pdf_to_text(input_path)
        
        if extracted_text and len(extracted_text.strip()) > 50:  # Minimum content check
            # Save to output file
            output_path = create_output_file(extracted_text, input_path, output_folder)
            if output_path:
                successful += 1
                print(f"[INFO] ✅ Successfully processed: {filename}")
            else:
                print(f"[ERROR] ❌ Failed to save output for: {filename}")
        else:
            print(f"[WARNING] ⚠️ Insufficient text extracted from: {filename}")

    print(f"\n[DONE] 🎉 Processing complete!")
    print(f"[DONE] Successfully processed: {successful}/{len(pdf_files)} files")
    print(f"[DONE] Output location: {output_folder}")

# --- Run ---
if __name__ == "__main__":
    sanitize_pdfs_advanced()
