# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from pathlib import Path
from PIL import Image
import io
import logging
from app.inference import get_model

# Setup
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Fix CORS issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model once at startup
model = None

@app.on_event("startup")
async def startup():
    global model
    try:
        model = get_model()
        logger.info(f"✅ Model loaded successfully! Classes: {model.classes}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model
    
    try:
        # Check if model is loaded
        if model is None:
            try:
                model = get_model()
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "detail": f"Model not loaded: {str(e)}"}
                )
        
        # Read file
        contents = await file.read()
        if len(contents) == 0:
            return JSONResponse(
                status_code=400,
                content={"success": False, "detail": "Empty file"}
            )
        
        # Open image
        try:
            image = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"success": False, "detail": f"Invalid image: {str(e)}"}
            )
        
        # Predict
        result = model.predict(image, threshold=0.85)
        
        return {
            "success": True,
            "prediction": result["label"],
            "confidence": round(result["confidence"] * 100, 2),
            "probabilities": result["probs"]
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "detail": str(e)}
        )

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes": model.classes if model else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
