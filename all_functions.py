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