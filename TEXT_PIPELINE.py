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

def run_text_pipeline(claim: str):
    retrieved_docs = retriver.retrieve_evidence(claim, faiss_index, evidence_corpus)
    reranked_docs = reranker.rerank_evidendce(claim, retrieved_docs)
    
    if not reranked_docs:
        return {"final_verdict": "NOT ENOUGH INFO", "explanation": "Could not find any relevant documents."}
    final_verdict, _ = classifier(claim, reranked_docs)
    top_evidence_for_summary = reranked_docs[:3]
    _, explanation = summarizer(claim, top_evidence_for_summary, final_verdict)
    best_evidence_text = reranked_docs[0][1]
    truncated_evidence = (best_evidence_text[:250] + '...') if len(best_evidence_text) > 250 else best_evidence_text

    clean_report = {
        "final_verdict": final_verdict,
        "explanation": explanation,
        "top_evidence_snippet": truncated_evidence
    }
    
    return clean_report

# --- AND UPDATE YOUR FINAL PRINT ---
if __name__ == "__main__":
    user_claim = "The Eiffel Tower is made of cheese."
    report = run_text_pipeline(user_claim)
    
    print("\n--- 🚀 FINAL TEXT ANALYSIS REPORT 🚀 ---")
    # This print block now works with the smaller, cleaner report
    print(f"Verdict: {report['final_verdict']}")
    print(f"Explanation: {report['explanation']}")
    print(f"Based on evidence: \"{report['top_evidence_snippet']}\"")