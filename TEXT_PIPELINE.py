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
from all_functions import reranker, retriver, classifier

retriver = retriver()
reranker = reranker()
classifier = classifier()

try:
    df= pd.read_csv('data.csv')
    evidence_corpus = df['evidence'].dropna().tolist()
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
    faiss_index = retriver.build_faiss_index(evidence_corpus)

def run_text_pipeline(claim):
    retrived_docs= retriver.retrieve_evidence(claim=claim, index=faiss_index, evidence_corpus=evidence_corpus)
    ranked_docs= reranker.rerank_evidendce(claim=claim, evidence_list=retrived_docs)
    if not ranked_docs:
        return {"verdict": "NOT ENOUGH INFO", "explanation": "Could not find any relevant documents in the knowledge base."}
    top_evidence = ranked_docs[0][1]
    final_report= classifier.get_final_verdict(claim=claim,top_evidence=top_evidence)

    return final_report


