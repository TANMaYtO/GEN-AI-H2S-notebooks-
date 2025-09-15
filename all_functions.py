import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import pipeline

class retriver:
    def __init__(self):
        self.retrivermodel= SentenceTransformer('all-MiniLM-L6-v2')
    def build_faiss_idx(self,evidence_corpus):
        embeddings= self.retrivermodel.encode(evidence_corpus, convert_to_tensor=True)
        index= faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.cpu().numpy())
        faiss.write_index(index, "evidence_index.faiss")
        return index

    def retrieve_evidence(self,claim, index, evidence_corpus, top_k=10):
        claim_embedding = self.retrivermodel.encode([claim])
        distances, indices = index.search(claim_embedding, top_k)
        retrieved_docs = [evidence_corpus[i] for i in indices[0]]
        return retrieved_docs
    
class reranker:
    def __init__(self):
        self.reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    def rerank_evidendce(self,claim, evidence_list):
        sentance_pairs= [[claim,evidence] for evidence in evidence_list]
        score= self.reranker_model.predict(sentance_pairs)
        scored_evidence= sorted(zip(score, evidence_list), reverse=True)
        return scored_evidence

class classifier:
    def __init__(self):
        self.summarizer = pipeline("text2text-generation", model="google/flan-t5-large")
    def result_parser(self,raw_text):
        try:
            lines = raw_text.strip().split('\n')
            verdict = lines[0].replace('VERDICT:', '').strip()
            explanation = lines[1].replace('EXPLANATION:', '').strip()
            return {
                'verdict': verdict,
                'explanation': explanation
            }
        except IndexError:
            return {
                'verdict': 'UNCLEAR',
                'explanation': raw_text.strip()
            }
    def get_final_verdict(self,claim, top_evidence):
        prompt = f"""
        You are a precise fact-checking AI. Your task is to analyze the trusted evidence and determine if it supports, refutes, or is insufficient to verify the user's claim.

        First, choose the single best label from the following options: [SUPPORTS, REFUTES, NOT ENOUGH INFO].
        Second, provide a concise, one-sentence explanation for your choice based on the evidence.

        Trusted Evidence: "{top_evidence}"

        User's Claim: "{claim}"

        Output your response in the following format:
        VERDICT: [Your chosen label]
        EXPLANATION: [Your one-sentence explanation]
        """
        result = self.summarizer(prompt, max_new_tokens=100)[0]['generated_text']
        return self.result_parser(result)

