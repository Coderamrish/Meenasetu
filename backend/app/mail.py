from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import shutil
from pathlib import Path
import uuid
from datetime import datetime
import os

# Import ML models
try:
    from .ml_models import meenasetu_models
    MODELS_LOADED = True
except Exception as e:
    print(f"⚠️ Failed to load ML models: {e}")
    MODELS_LOADED = False

# Initialize FastAPI app
app = FastAPI(
    title="MeenaSetu AI API",
    description="Fish Species Classification & Disease Detection API",
    version="1.0.0"
)

# Configure CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to MeenaSetu AI API",
        "status": "running",
        "models_loaded": MODELS_LOADED,
        "endpoints": {
            "/": "API documentation (this page)",
            "/health": "Health check",
            "/status": "Model status",
            "/upload": "Upload and process image (POST)",
            "/docs": "Interactive API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": MODELS_LOADED,
        "uploads_dir": str(UPLOADS_DIR.absolute())
    }

@app.get("/status")
async def get_status():
    """Get model loading status"""
    if MODELS_LOADED:
        return meenasetu_models.get_status()
    else:
        return {
            "species_loaded": False,
            "disease_loaded": False,
            "message": "Models failed to load"
        }

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload and process fish image for species classification and disease detection
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Generate unique filename
        file_extension = Path(file.filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = UPLOADS_DIR / unique_filename
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process image with ML models
        if MODELS_LOADED:
            result = meenasetu_models.process_image(str(file_path))
        else:
            result = {
                "status": "models_not_loaded",
                "message": "ML models failed to load on server startup"
            }
        
        # Add file info to result
        result["filename"] = file.filename
        result["uploaded_filename"] = unique_filename
        result["file_size"] = file_path.stat().st_size
        result["content_type"] = file.content_type
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-upload")
async def test_upload_page():
    """Simple test page for uploads"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MeenaSetu AI - Test Upload</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
            .result { margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>🐟 MeenaSetu AI Test</h1>
        <p>Upload a fish image for species and disease analysis</p>
        
        <div class="upload-area">
            <input type="file" id="imageInput" accept="image/*">
            <br><br>
            <button onclick="uploadImage()">Analyze Image</button>
        </div>
        
        <div id="result" class="result" style="display: none;"></div>
        
        <script>
            async function uploadImage() {
                const fileInput = document.getElementById('imageInput');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select an image first');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = '<p>Processing...</p>';
                resultDiv.style.display = 'block';
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    let html = '<h3>Results:</h3>';
                    
                    if (data.species_detection) {
                        html += `<p><strong>Species:</strong> ${data.species_detection.predicted_species || 'Unknown'}</p>`;
                        html += `<p><strong>Confidence:</strong> ${(data.species_detection.confidence * 100).toFixed(1)}%</p>`;
                    }
                    
                    if (data.disease_detection) {
                        html += `<p><strong>Health:</strong> ${data.disease_detection.is_healthy ? 'Healthy 🟢' : 'Disease 🔴'}</p>`;
                        html += `<p><strong>Disease:</strong> ${data.disease_detection.predicted_disease || 'Unknown'}</p>`;
                    }
                    
                    resultDiv.innerHTML = html;
                    
                } catch (error) {
                    resultDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)