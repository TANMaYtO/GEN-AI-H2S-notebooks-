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
from pmo_func import reranker, retriver, Classifier, summarizer,FactChecker
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
    retrieved_docs,indices = retriver.retrieve_evidence(claim, faiss_index, evidence_corpus)
    reranked_docs = reranker.rerank_evidendce(claim, retrieved_docs)
    
    if not reranked_docs:
        try:
            google = FactChecker()
            result,arc = google.check_claim(claim)
            return {
                "final verdict":result['verdict'],
                "explanation":result['summary'],
                "source":{a['source']:a['url'] for a in arc}
            }
        except Exception as e:
            print(f"Google Could not help and i dont think even god can | {e}")

    final_verdict, _ = classifier(claim, reranked_docs)
    top_evidence_for_summary = reranked_docs[:3]
    _, explanation = summarizer(claim, top_evidence_for_summary, final_verdict)
    best_evidence_text = reranked_docs[0][1]
    # truncated_evidence = (best_evidence_text[:250] + '...') if len(best_evidence_text) > 250 else best_evidence_text
    df_rel = df.iloc[indices]
    sources_dict = df_rel.set_index("source")["url"].to_dict()
    
    clean_report = {
        "final_verdict": final_verdict,
        "explanation": explanation,
        "source": sources_dict
    }
    return clean_report

# --- AND UPDATE YOUR FINAL PRINT ---
if __name__ == "__main__":
    user_claim = "The Indian president is Narendra Modi."
    report = run_text_pipeline(user_claim)
    
    print("\n--- 🚀 FINAL TEXT ANALYSIS REPORT 🚀 ---")
    # This print block now works with the smaller, cleaner report
    print(f"Verdict: {report['final_verdict']}")
    print(f"Explanation: {report['explanation']}")
    print(f"Based on evidence: \"{report['top_evidence_snippet']}\"")