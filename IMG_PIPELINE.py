import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import pipeline
from PIL import Image, ImageChops, ImageEnhance
import torch
from google.cloud import vision
import os
import io
from all_functions import img_manipulation, OCR
from TEXT_PIPELINE import run_text_pipeline
import json

manipulation_analyzer = img_manipulation()
ocr_analyzer = OCR()

def run_img_pipeline(img_pth):
    manipulation_results = manipulation_analyzer.run_image_forensics(img_pth)
    in_image_report = ocr_analyzer.get_in_image_anal(img_pth)
    rev_img_search_res = ocr_analyzer.rev_img_search(img_pth)
    
    text_analysis_report = {}
    if in_image_report.get("Extracted Text", "").strip():
        text_analysis_report = run_text_pipeline(in_image_report["Extracted Text"])
    final_report = {
        'image_manipulation_report': manipulation_results,
        'in_image_content_report': in_image_report,
        'reverse_image_search_report': rev_img_search_res,
        'extracted_text_analysis_report': text_analysis_report
    }
    return final_report


if __name__ == "__main__":
    # 1. Provide the path to an image you want to test
    #    Use an image with text for the best test, like a news screenshot.
    test_image_path = r"E:\GENAI H2S\WhatsApp Image 2025-09-17 at 23.22.10_7bd3fcb6.jpg"
    
    # 2. Check if the file exists before running
    if os.path.exists(test_image_path):
        # 3. Call your main image pipeline function
        final_report = run_img_pipeline(test_image_path)
        
        # 4. Pretty-print the complete, consolidated report
        print("\n\n--- 🚀 FINAL IMAGE ANALYSIS REPORT 🚀 ---")
        print(json.dumps(final_report, indent=2))
    else:
        print(f"🔴 Error: Test image not found at '{test_image_path}'")
