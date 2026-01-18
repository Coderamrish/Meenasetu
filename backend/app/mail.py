from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import shutil
from pathlib import Path
import uuid
from datetime import datetime
import os
import traceback
import sys

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
            "/diagnostics": "System diagnostics",
            "/upload": "Upload and process image (POST)",
            "/test-models": "Test models with detailed output (POST)",
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

@app.get("/diagnostics")
async def run_diagnostics():
    """
    Comprehensive system diagnostics
    """
    diagnostics = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "models_loaded": MODELS_LOADED,
        "uploads_dir": str(UPLOADS_DIR.absolute()),
        "uploads_dir_exists": UPLOADS_DIR.exists(),
        "uploads_dir_writable": UPLOADS_DIR.exists() and os.access(UPLOADS_DIR, os.W_OK),
    }
    
    # Check imports
    import_status = {}
    required_packages = {
        'PIL': 'Pillow',
        'numpy': 'numpy',
        'tensorflow': 'tensorflow',
        'torch': 'torch',
        'cv2': 'opencv-python'
    }
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            import_status[package] = {"status": "✓ Installed", "pip_name": pip_name}
        except ImportError:
            import_status[package] = {"status": "✗ Not Found", "pip_name": pip_name}
    
    diagnostics["packages"] = import_status
    
    # Check ML models if loaded
    if MODELS_LOADED:
        try:
            from .ml_models import meenasetu_models
            model_status = meenasetu_models.get_status()
            diagnostics["models"] = model_status
        except Exception as e:
            diagnostics["models"] = {"error": str(e)}
    else:
        diagnostics["models"] = {"status": "not_loaded"}
    
    return diagnostics

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload and process fish image for species classification and disease detection
    WITH ENHANCED ERROR HANDLING AND DIAGNOSTICS
    """
    file_path = None
    
    try:
        # Step 1: Validate file type
        print(f"\n{'='*60}")
        print(f"[UPLOAD] Starting image upload process")
        print(f"[UPLOAD] Filename: {file.filename}")
        print(f"[UPLOAD] Content-Type: {file.content_type}")
        print(f"{'='*60}\n")
        
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail=f"File must be an image. Got: {file.content_type}"
            )
        
        # Step 2: Generate unique filename and save
        file_extension = Path(file.filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = UPLOADS_DIR / unique_filename
        
        print(f"[SAVE] Saving to: {file_path}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = file_path.stat().st_size
        print(f"[SAVE] ✓ File saved successfully ({file_size} bytes)")
        
        # Step 3: Verify file exists and is readable
        if not file_path.exists():
            raise Exception(f"File was not saved correctly: {file_path}")
        
        print(f"[VERIFY] ✓ File exists and is readable")
        
        # Step 4: Process with ML models
        print(f"[MODELS] Starting ML model processing...")
        print(f"[MODELS] Models loaded: {MODELS_LOADED}")
        
        if not MODELS_LOADED:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "models_not_loaded",
                    "message": "ML models failed to load on server startup",
                    "filename": file.filename,
                    "uploaded_filename": unique_filename,
                    "file_size": file_size,
                    "content_type": file.content_type
                }
            )
        
        # Process the image
        print(f"[PROCESS] Calling process_image()...")
        try:
            result = meenasetu_models.process_image(str(file_path))
            print(f"[PROCESS] ✓ Image processing completed successfully")
            print(f"[RESULT] Keys in result: {result.keys()}")
        except Exception as process_error:
            print(f"[ERROR] Image processing failed: {process_error}")
            traceback.print_exc()
            raise Exception(f"Image processing failed: {str(process_error)}")
        
        # Step 5: Add file info to result
        result["filename"] = file.filename
        result["uploaded_filename"] = unique_filename
        result["file_size"] = file_size
        result["content_type"] = file.content_type
        
        print(f"\n{'='*60}")
        print(f"[SUCCESS] Upload and analysis completed successfully")
        print(f"{'='*60}\n")
        
        return JSONResponse(status_code=200, content=result)
        
    except HTTPException as he:
        # Re-raise HTTP exceptions as-is
        print(f"[HTTP_ERROR] {he.detail}")
        raise he
        
    except Exception as e:
        # Detailed error logging
        error_type = type(e).__name__
        error_message = str(e)
        error_traceback = traceback.format_exc()
        
        print(f"\n{'!'*60}")
        print(f"[CRITICAL ERROR] Unhandled exception in upload endpoint")
        print(f"[ERROR TYPE] {error_type}")
        print(f"[ERROR MESSAGE] {error_message}")
        print(f"[TRACEBACK]")
        print(error_traceback)
        print(f"{'!'*60}\n")
        
        # Clean up file if it was created
        if file_path and file_path.exists():
            try:
                file_path.unlink()
                print(f"[CLEANUP] Removed failed upload file: {file_path}")
            except:
                pass
        
        # Return detailed error to client
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error_type": error_type,
                "error_message": error_message,
                "traceback": error_traceback.split('\n'),
                "suggestions": [
                    "Check if ML models are properly initialized",
                    "Verify image file is valid and not corrupted",
                    "Ensure all required packages are installed (PIL, numpy, tensorflow, torch)",
                    "Check model file paths and permissions",
                    "Review the traceback above for specific error location"
                ]
            }
        )

@app.post("/test-models")
async def test_models(file: UploadFile = File(...)):
    """
    Test ML models with detailed step-by-step output
    """
    results = {
        "steps": [],
        "errors": [],
        "success": False
    }
    
    file_path = None
    
    try:
        # Save file
        results["steps"].append("Saving uploaded file...")
        file_extension = Path(file.filename).suffix
        unique_filename = f"test_{uuid.uuid4()}{file_extension}"
        file_path = UPLOADS_DIR / unique_filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        results["steps"].append(f"✓ File saved: {file_path}")
        
        # Load image
        results["steps"].append("Loading image with PIL...")
        from PIL import Image
        img = Image.open(file_path)
        results["steps"].append(f"✓ Image loaded: {img.size} pixels, mode: {img.mode}")
        
        # Test species model
        if MODELS_LOADED:
            results["steps"].append("Testing species classification...")
            try:
                species_result = meenasetu_models.classify_species(str(file_path))
                results["steps"].append(f"✓ Species: {species_result}")
            except AttributeError:
                results["steps"].append("⚠ classify_species() method not found, trying process_image()...")
                try:
                    full_result = meenasetu_models.process_image(str(file_path))
                    results["steps"].append(f"✓ Full result: {full_result}")
                except Exception as e:
                    results["errors"].append(f"process_image() error: {str(e)}")
            except Exception as e:
                results["errors"].append(f"Species classification error: {str(e)}")
                results["errors"].append(traceback.format_exc())
            
            # Test disease model
            results["steps"].append("Testing disease detection...")
            try:
                disease_result = meenasetu_models.detect_disease(str(file_path))
                results["steps"].append(f"✓ Disease: {disease_result}")
            except AttributeError:
                results["steps"].append("⚠ detect_disease() method not found (using process_image() above)")
            except Exception as e:
                results["errors"].append(f"Disease detection error: {str(e)}")
                results["errors"].append(traceback.format_exc())
        else:
            results["errors"].append("Models not loaded")
        
        results["success"] = len(results["errors"]) == 0
        
    except Exception as e:
        results["errors"].append(f"Critical error: {str(e)}")
        results["traceback"] = traceback.format_exc()
    
    finally:
        # Cleanup
        if file_path and file_path.exists():
            file_path.unlink()
            results["steps"].append("✓ Cleaned up test file")
    
    return results

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