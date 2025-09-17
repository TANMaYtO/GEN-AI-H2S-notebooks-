import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import pipeline
from PIL import Image, ImageChops, ImageEnhance
import torch
from google.cloud import vision
import os
import io
from all_functions import reranker, retriver, Classifier, summarizer
import json

retriver = retriver()
reranker = reranker()
classifier = Classifier()
summarizer= summarizer()

try:
    df= pd.read_csv('data.csv')
    evidence_corpus = df['text'].dropna().tolist()
except FileNotFoundError:
    print("your_dataset.csv not found! Please check the filename.")
    evidence_corpus = []
except KeyError:
    print("Column 'evidence_column_name' not found in the CSV. Please check the column name.")
    evidence_corpus = []

index_file = "evidence_index.faiss"
if os.path.exists(index_file):
    faiss_index = faiss.read_index(index_file)
else:
    faiss_index = retriver.build_faiss_idx(evidence_corpus)

def run_text_pipeline(claim):
    retrieved_docs= retriver.retrieve_evidence(claim=claim, index= faiss_index, evidence_corpus=evidence_corpus)
    reranked_docs = reranker.rerank_evidendce(claim=claim, evidence_list=retrieved_docs)
    if not reranked_docs:
        return {"verdict": "NOT ENOUGH INFO", "explanation": "Could not find any relevant documents.", "evidence_report": []}
    final_verdict, detailed_verdicts = classifier(claim, reranked_docs)
    _, explanation = summarizer(claim, reranked_docs[:3], final_verdict)
    final_report = {
        "final_verdict": final_verdict,
        "explanation": explanation,
        "evidence_report": detailed_verdicts
    }
    return final_report



if __name__ == "__main__":
    user_claim = "The Eiffel Tower is made of cheese."
    
    report = run_text_pipeline(user_claim)
    
    print("\n--- 🚀 FINAL TEXT ANALYSIS REPORT 🚀 ---")
    # Use json.dumps for a clean, readable print of the nested dictionary
    print(json.dumps(report, indent=2))