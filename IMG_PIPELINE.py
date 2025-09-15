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

def run_img_pipeline(img_pth):
    manipulation_results= img_manipulation.run_image_forensics(img_pth)
    in_image_report= OCR.get_in_image_anal(img_pth)
    run_text_pipeline(in_image_report['Extracted Text'])
    rev_img_search_res= OCR.rev_img_search(img_pth)
    final_report= {
        'image manipulation result': manipulation_results,
        'in image report': in_image_report,
        'reverse image search result': rev_img_search_res
    }
    return final_report
