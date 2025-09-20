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
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import trafilatura as tra

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class retriver:
    def __init__(self):
        self.retrivermodel = SentenceTransformer('all-MiniLM-L6-v2')

    def build_faiss_idx(self, evidence_corpus):
        embeddings = self.retrivermodel.encode(evidence_corpus)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(np.array(embeddings, dtype=np.float32))
        faiss.write_index(index, "evidence_index.faiss")
        return index

    def retrieve_evidence(self, claim, index, evidence_corpus, top_k=10):
        claim_embedding = self.retrivermodel.encode([claim])
        distances, indices = index.search(np.array(claim_embedding, dtype=np.float32), top_k)
        retrieved_docs = [evidence_corpus[i] for i in indices[0]]
        return retrieved_docs, indices[0]

class reranker:
    def __init__(self):
        self.reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=DEVICE)

    def rerank_evidendce(self, claim, evidence_list):
        sentance_pairs = [[claim, evidence] for evidence in evidence_list]
        score = self.reranker_model.predict(sentance_pairs)
        scored_evidence = sorted(zip(score, evidence_list), reverse=True)
        return scored_evidence

class Classifier:
    def __init__(self):
        self.model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        self.label_names = ["entailment", "neutral", "contradiction"]
        self.device = torch.device(DEVICE)
        print(f"Classifier device: {self.device}")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.eval()

    def classify(self, claim, top_evidence):
        verdicts = []
        evidences = [e[1] for e in top_evidence]
        if not evidences:
            return "NEUTRAL", []
        
        inputs = self.tokenizer(evidences, [claim] * len(evidences), return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=-1)
        for i, evidence in enumerate(evidences):
            pred = torch.argmax(probs[i]).item()
            verdicts.append({
                "evidence": evidence,
                "verdict": self.label_names[pred],
                "scores": {name: float(probs[i][j]) for j, name in enumerate(self.label_names)}
            })

        top_verdict_info = verdicts[0]
        if top_verdict_info["verdict"] == "entailment" and top_verdict_info["scores"]["entailment"] > 0.8:
            result = "TRUE"
        elif top_verdict_info["verdict"] == "contradiction" and top_verdict_info["scores"]["contradiction"] > 0.8:
            result = "FALSE"
        else:
            for v in verdicts[1:]:
                if v["verdict"] == "contradiction" and v["scores"]["contradiction"] > 0.9:
                    result = "FALSE"
                    break
            else:
                result = "NEUTRAL"
        return result, verdicts

    def __call__(self, claim, evidences):
        return self.classify(claim, evidences)

class summarizer:
    def __init__(self):
        self.model_name = "google/flan-t5-base" # Using a smaller model for server efficiency
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.device = torch.device(DEVICE)
        self.model.to(self.device)
        self.model.eval()
        print(f"Summarizer device: {self.device}")

    def forward(self, claim, top_evidence, verdict, max_input_len=1024, max_output_len=150):
        evidence_texts = [e[1] for e in top_evidence]
        if not evidence_texts:
            return verdict, "No evidence was provided to generate a summary."
        
        input_text = f"""Claim: "{claim}"\nVerdict: {verdict}\nEvidence:\n{"\n---\n".join(evidence_texts)}\n\nWrite a short, neutral explanation for why the verdict is {verdict}, based only on the evidence provided."""
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_input_len).to(self.device)
        
        with torch.no_grad():
            summary_ids = self.model.generate(inputs["input_ids"], max_length=max_output_len, num_beams=4, early_stopping=True)
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return verdict, summary

    def __call__(self, claim, top_evidence, verdict):
        return self.forward(claim, top_evidence, verdict)

class FactChecker:
    def __init__(self):
        self.factcheck_api = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        self.google_search = "https://www.google.com/search"
        load_dotenv()
        self.factcheck_api_key = os.getenv("GOOGLE_FACT_CHECK_API_KEY")

    def check_google_factcheck(self, claim: str, pages: int = 5):
        if not self.factcheck_api_key: return None
        params = {'key': self.factcheck_api_key, 'query': claim, 'languageCode': 'en-US', 'pageSize': pages}
        try:
            response = requests.get(self.factcheck_api, params=params)
            response.raise_for_status()
            data = response.json()
            if 'claims' in data and data['claims']:
                review = data['claims'][0].get('claimReview', [{}])[0]
                return {"verdict": review.get('textualRating', 'Unknown'), "summary": f"Rated by {review.get('publisher', {}).get('name', 'Unknown')}"}
        except Exception as e:
            print(f"FactCheck API error: {e}")
        return None

    # Note: Web scraping is fragile. The rest of your web search logic would go here.
    # For this refactor, we'll assume the RAG pipeline is the primary method.
    def check_claim(self, claim: str):
        print(f"\n--- Checking claim via Google Fact Check API: '{claim}' ---")
        return self.check_google_factcheck(claim)

class img_manipulation:
    def __init__(self):
        self.GEN_AI_IMAGE = pipeline("image-classification", model="umm-maybe/AI-image-detector", device=DEVICE)

    def Gen_AI_IMG(self, img_pth):
        try:
            with Image.open(img_pth) as img:
                img = img.convert('RGB')
                result = self.GEN_AI_IMAGE(img)
            proba = next((item['score'] for item in result if item['label'] == 'artificial'), 0.0)
            return proba * 100
        except Exception as e:
            print(f'AI image detection error: {e}')
            return 0.0

    def generated_image(self, img_pth, quality=90, scale=15):
        try:
            with Image.open(img_pth) as orig_img:
                orig_img = orig_img.convert('RGB')
                temp_path = 'temp_resaved.jpg'
                orig_img.save(temp_path, 'JPEG', quality=quality)
                with Image.open(temp_path) as resaved_img:
                    ela_image = ImageChops.difference(orig_img, resaved_img)
            os.remove(temp_path)
            ela_data = np.array(ela_image)
            mean_intensity = ela_data.mean()
            scaled_score = min(100, (mean_intensity / 25.0) * 100)
            
            # Save the ELA image and return its path for serving
            ela_path = "ela_result.png"
            enhancer = ImageEnhance.Brightness(ela_image)
            max_diff = max(1, max([ex[1] for ex in ela_image.getextrema()]))
            ela_image_enhanced = enhancer.enhance(scale / max_diff)
            ela_image_enhanced.save(ela_path)
            return scaled_score, ela_path
        except Exception as e:
            print(f'ELA generation error: {e}')
            return 0.0, None

    def run_image_forensics(self, image_path):
        ai_score = self.Gen_AI_IMG(image_path)
        classic_score, ela_path = self.generated_image(image_path)
        return {
            "ai_generated_score_percent": ai_score,
            "classic_edit_score_percent": classic_score,
            "ela_image_path": ela_path
        }

class OCR:
    def __init__(self, key_path='GOOGLE_VISION_API.json'):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path
        self.client = vision.ImageAnnotatorClient()

    def _get_full_vision_analysis(self, img_pth):
        try:
            with open(img_pth, 'rb') as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            features = [{'type_': vision.Feature.Type.DOCUMENT_TEXT_DETECTION}, {'type_': vision.Feature.Type.SAFE_SEARCH_DETECTION}, {'type_': vision.Feature.Type.LANDMARK_DETECTION}, {'type_': vision.Feature.Type.LOGO_DETECTION}, {'type_': vision.Feature.Type.WEB_DETECTION}]
            response = self.client.annotate_image({'image': image, 'features': features})
            return response, None
        except Exception as e:
            return None, str(e)

    def get_in_image_anal(self, img_pth):
        response, error = self._get_full_vision_analysis(img_pth)
        if error: return {'error': error}
        report = {}
        if response.full_text_annotation: report['Extracted Text'] = response.full_text_annotation.text
        if response.safe_search_annotation:
            safe = response.safe_search_annotation
            report['Safe Search'] = {'adult': vision.Likelihood(safe.adult).name, 'violence': vision.Likelihood(safe.violence).name}
        entities = []
        if response.landmark_annotations: entities.extend([f'Landmark: {l.description}' for l in response.landmark_annotations])
        if response.logo_annotations: entities.extend([f'Logo: {l.description}' for l in response.logo_annotations])
        if entities: report['Identified Entities'] = entities
        return report

    def rev_img_search(self, img_pth):
        response, error = self._get_full_vision_analysis(img_pth)
        if error: return {'error': error}
        report = {}
        if response.web_detection and response.web_detection.pages_with_matching_images:
            matches = [{'title': p.page_title, 'url': p.url} for p in response.web_detection.pages_with_matching_images[:5]]
            report['Reverse Image Matches'] = matches
        return report

