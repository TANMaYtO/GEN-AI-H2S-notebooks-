import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import pipeline

retrivermodel= SentenceTransformer('all-MiniLM-L6-v2')
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
summarizer = pipeline("text2text-generation", model="google/flan-t5-large")

def build_faiss_idx(evidence_corpus):
    embeddings= retrivermodel.encode(evidence_corpus, convert_to_tensor=True)
    index= faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.cpu().numpy())
    faiss.write_index(index, "evidence_index.faiss")
    return index

def retrieve_evidence(claim, index, evidence_corpus, top_k=10):
    claim_embedding = retrivermodel.encode([claim])
    distances, indices = index.search(claim_embedding, top_k)
    retrieved_docs = [evidence_corpus[i] for i in indices[0]]
    return retrieved_docs

