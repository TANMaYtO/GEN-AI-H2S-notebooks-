import pandas as pd
import faiss
import os
import json

def run_text_pipeline(claim: str, state: dict):
    """
    Executes the text analysis pipeline using pre-loaded tools.
    """
    # Unpack all the necessary tools and data from the state dictionary
    retriever = state['retriever']
    reranker = state['reranker']
    classifier = state['classifier']
    summarizer = state['summarizer']
    fact_checker = state['fact_checker']
    df = state['df']
    evidence_corpus = state['evidence_corpus']
    faiss_index = state['faiss_index']

    # --- RAG Pipeline ---
    retrieved_docs, indices = retriever.retrieve_evidence(claim, faiss_index, evidence_corpus)
    reranked_docs = reranker.rerank_evidendce(claim, retrieved_docs)
    
    if not reranked_docs:
        # --- Fallback to Google Fact Check ---
        print("No results from RAG, trying Google Fact Check...")
        result = fact_checker.check_claim(claim)
        return {
            "final_verdict": result.get('verdict', 'NEUTRAL'),
            "explanation": result.get('summary', 'Could not verify claim.'),
            "source": {result.get('source'): result.get('URLs', ['#'])[0]} if result else {}
        }

    final_verdict, _ = classifier(claim, reranked_docs)
    top_evidence_for_summary = reranked_docs[:3]
    _, explanation = summarizer(claim, top_evidence_for_summary, final_verdict)
    
    # Get sources from the original dataframe
    sources_dict = {}
    if len(indices) > 0 and 'source' in df.columns and 'url' in df.columns:
        df_rel = df.iloc[indices]
        # Handle potential duplicate sources by taking the first URL for each source
        sources_dict = df_rel.groupby('source')['url'].first().to_dict()

    return {
        "final_verdict": final_verdict,
        "explanation": explanation,
        "source": sources_dict
    }

