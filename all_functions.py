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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

class Classifier:
    def __init__(self):
        self.model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        self.label_names = ["entailment", "neutral", "contradiction"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Could not fetch model from Hugging Face | {e}")

    def classify(self, claim, top_evidence):     
        self.verdicts = []  #
        evidences = [e[1] for e in top_evidence]
        if not evidences:
            raise ValueError("No evidence provided")
        try:
            inputs = self.tokenizer(
                evidences,
                [claim] * len(evidences),
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            with torch.no_grad():
                inputs = {k:v.to(self.device) for k,v in inputs.items()}
                outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            for i, evidence in enumerate(evidences):
                pred = torch.argmax(probs[i]).item()
                self.verdicts.append({
                    "evidence": evidence,
                    "verdict": self.label_names[pred],
                    "scores": {name: float(probs[i][j]) for j, name in enumerate(self.label_names)}
                })
            labels = [v["verdict"] for v in self.verdicts]
            if "entailment" in labels:
                result = "TRUE"
            elif "contradiction" in labels:
                result = "FALSE"
            else:
                result = "NEUTRAL"

            return result, self.verdicts
        except Exception as e:
            raise RuntimeError(f"Classification failed | {e}") 
    def __call__(self,claim,evidences):
        return self.classify(claim,evidences)

    
class img_manipulation:
    def __init__(self):
        self.GEN_AI_IMAGE = pipeline("image-classification", model="umm-maybe/AI-image-detector", device = DEVICE)
    def Gen_AI_IMG(self,img_pth):
        try:
            img = Image.open(img_pth).convert('RGB')
            result = self.GEN_AI_IMAGE(img)
            manipulation_proba = 0
            for i in result:
                if i['label'] == 'artificial':
                    manipulation_proba = i['score']
                    break
            manipulation_proba = manipulation_proba*100
            print(f'chance of manipulation: {manipulation_proba:.2f}')
            return manipulation_proba
        except Exception as e:
            print(f'an error occured:{e}')
            return None
    def generated_image(self,img_pth, quality= 90, scale= 15):
        try:
            orig_img = Image.open(img_pth).convert('RGB')
            temp_path = 'temp_resaved.jpg'
            orig_img.save(temp_path, 'JPEG', quality=quality)
            resaved_img = Image.open(temp_path)
            ela_image= ImageChops.difference(orig_img, resaved_img)
            ela_data = np.array(ela_image)
            mean_intensity = ela_data.mean()
            scaled_score = min(100, (mean_intensity / 25.0) * 100)
            extrema= ela_image.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            if max_diff == 0:
                max_diff = 1
            enhancer = ImageEnhance.Brightness(ela_image)
            ela_image = enhancer.enhance(scale / max_diff)
            print(f"ELA image generated for {img_pth}")
            ela_image.show()
            return scaled_score, ela_image
        except Exception as e:
            print(f'an error occured: {e}')
            return None
    def run_image_forensics(self,image_path):
        ai_generated_score = self.Gen_AI_IMG(image_path)
        classic_edit_score, ela_image_obj = self.generated_image(image_path)
        
        results = {
            "ai_generated_score_percent": f'{ai_generated_score:.2f}',
            "classic_edit_score_percent": f'{classic_edit_score:.2f}',
            "ela_image": ela_image_obj
        }
        print(f"AI-Generated Score: {results['ai_generated_score_percent']}%")
        print(f"Classic Edit Score (ELA): {results['classic_edit_score_percent']}%")
        
        if results["ela_image"]:
            print("ELA image has been generated and is available for display.")  
        return results

class OCR:
    def __init__(self):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'GOOGLE_VISION_API.json'
        self.client = vision.ImageAnnotatorClient()
    def get_full_vision_anal(self,img_pth):
        try:
            with open(img_pth, 'rb') as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            features = [
                {'type_': vision.Feature.Type.DOCUMENT_TEXT_DETECTION},
                {'type_': vision.Feature.Type.SAFE_SEARCH_DETECTION},
                {'type_': vision.Feature.Type.LANDMARK_DETECTION},
                {'type_': vision.Feature.Type.LOGO_DETECTION},
                {'type_': vision.Feature.Type.WEB_DETECTION}
            ]
            response = self.client.annotate_image({'image': image, 'features': features})
            return response, None
        except Exception as e:
            return None, str(e)
    def get_in_image_anal(self,img_pth):
        response, error = self.get_full_vision_anal(img_pth)
        if error:
            return {'error': error}
        report = {}
        # OCR
        if response.full_text_annotation:
            report['Extracted Text'] = response.full_text_annotation.text
        # SAFE SEARCH
        if response.safe_search_annotation:
            safe = response.safe_search_annotation
            report['Safe Search'] = {
                'adult': vision.Likelihood(safe.adult).name,
                'violence': vision.Likelihood(safe.violence).name,
                'spoof': vision.Likelihood(safe.spoof).name
            }
        # LANDMARKS AND LOGOS
        entities = []
        if response.landmark_annotations:
            for landmark in response.landmark_annotations:
                entities.append(f'Landmark: {landmark.description}')
        if response.logo_annotations:
            for logo in response.logo_annotations:
                entities.append(f'Logo: {logo.description}')
        if entities:
            report['Identified Entities'] = entities
        return report
    def rev_img_search(self,img_pth):
        response, error = self.get_full_vision_anal(img_pth)
        if error:
            return {'error': error}
        report = {}
        if response.web_detection and response.web_detection.pages_with_matching_images:
            matches = []
            for i in response.web_detection.pages_with_matching_images[:5]:
                matches.append({'title': i.page_title, 'url': i.url})
            report['Reverse Image Matches'] = matches
        return report