from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import os
import shutil
import pandas as pd
import faiss
# Import your classes and pipeline functions
from pmo_func import retriver, reranker, Classifier, summarizer, img_manipulation, OCR, FactChecker
from TEXT_PIPELINE import run_text_pipeline
from IMG_PIPELINE import run_img_pipeline

# This dictionary will hold all our initialized models and data
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads all models and data once when the server starts up."""
    print("--- 🚀 Server starting up... Loading all models... 🚀 ---")
    app_state['retriever'] = retriver()
    app_state['reranker'] = reranker()
    app_state['classifier'] = Classifier()
    app_state['summarizer'] = summarizer()
    app_state['manipulation_analyzer'] = img_manipulation()
    app_state['ocr_analyzer'] = OCR()
    app_state['fact_checker'] = FactChecker()
    
    try:
        df = pd.read_csv('data.csv', low_memory=False)
        app_state['evidence_corpus'] = df['text'].dropna().tolist()
        app_state['df'] = df
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load data.csv: {e}")
        app_state['evidence_corpus'] = []
        app_state['df'] = pd.DataFrame()

    index_file = "evidence_index.faiss"
    if os.path.exists(index_file):
        app_state['faiss_index'] = faiss.read_index(index_file)
    elif app_state['evidence_corpus']:
        print("Building FAISS index for the first time...")
        app_state['faiss_index'] = app_state['retriever'].build_faiss_idx(app_state['evidence_corpus'])
    else:
        app_state['faiss_index'] = None
    
    print("--- ✅ All models and data loaded successfully! ✅ ---")
    yield
    print("--- Shutting down ---")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (fine for a hackathon)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mounts the 'frontend_by_gemini' folder at the '/static' URL path
app.mount("/static", StaticFiles(directory="frontend_by_gemini"), name="static")
# Mounts the root directory to serve files like 'ela_result.png'
app.mount("/results", StaticFiles(directory="."), name="results")

@app.get("/")
async def read_index():
    return FileResponse('frontend_by_gemini/index.html')

@app.post("/analyze")
async def analyze_content(
    text_input: str = Form(None), 
    image_file: UploadFile = File(None)
):
    # This logic correctly prioritizes the image if both are sent
    if image_file and image_file.filename:
        try:
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, image_file.filename)
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(image_file.file, buffer)
            
            report = run_img_pipeline(temp_path, app_state)
            shutil.rmtree(temp_dir)
            return JSONResponse(content=report)
        except Exception as e:
            print(f"Error in image pipeline: {e}")
            raise HTTPException(status_code=500, detail="Error processing image.")
    
    elif text_input:
        try:
            report = run_text_pipeline(text_input, app_state)
            return JSONResponse(content=report)
        except Exception as e:
            print(f"Error in text pipeline: {e}")
            raise HTTPException(status_code=500, detail="Error processing text.")
            
    else:
        raise HTTPException(status_code=400, detail="No valid input provided.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), reload=True)