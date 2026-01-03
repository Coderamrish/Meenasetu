from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import shutil
import logging
from datetime import datetime
import sys

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import your MeenaSetu AI
try:
    from app.rag.chain import MeenasetuAI, Config
except ImportError:
    # Alternative import for direct script execution
    from rag.chain import MeenasetuAI, Config

# ============================================================
# 📋 LOGGING SETUP
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# 🚀 FASTAPI APP INITIALIZATION
# ============================================================

# Initialize MeenaSetu AI (singleton)
meenasetu_ai: Optional[MeenasetuAI] = None

# Modern lifespan event handler (replaces deprecated on_event)
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    global meenasetu_ai
    # Startup
    try:
        logger.info("🚀 Initializing MeenaSetu AI...")
        meenasetu_ai = MeenasetuAI()
        logger.info("✅ MeenaSetu AI ready!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize AI: {e}")
        raise
    
    yield  # Application is running
    
    # Shutdown
    logger.info("👋 Shutting down MeenaSetu AI...")
    meenasetu_ai = None

# Create FastAPI app with lifespan
app = FastAPI(
    title="🐠 MeenaSetu AI API",
    description="Intelligent Aquatic Biodiversity Expert - RAG, ML Classification & Data Analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MeenaSetu AI (singleton)
meenasetu_ai: Optional[MeenasetuAI] = None

# Modern lifespan event handler (replaces deprecated on_event)
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    global meenasetu_ai
    # Startup
    try:
        logger.info("🚀 Initializing MeenaSetu AI...")
        meenasetu_ai = MeenasetuAI()
        logger.info("✅ MeenaSetu AI ready!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize AI: {e}")
        raise
    
    yield  # Application is running
    
    # Shutdown
    logger.info("👋 Shutting down MeenaSetu AI...")
    meenasetu_ai = None

# Update app initialization to use lifespan
app = FastAPI(
    title="🐠 MeenaSetu AI API",
    description="Intelligent Aquatic Biodiversity Expert - RAG, ML Classification & Data Analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# ============================================================
# 📦 PYDANTIC MODELS
# ============================================================
class QueryRequest(BaseModel):
    query: str = Field(..., description="Question to ask the AI", min_length=1)
    include_sources: bool = Field(True, description="Include source documents in response")

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    source_count: int
    timestamp: str
    has_ml_classification: bool

class FileUploadResponse(BaseModel):
    status: str
    file: str
    file_type: str
    chunks_created: int
    ml_classified: bool
    message: str

class MultipleFilesUploadResponse(BaseModel):
    status: str
    total_files: int
    successful: int
    failed: int
    details: List[Dict[str, Any]]

class VisualizationRequest(BaseModel):
    plot_type: str = Field(..., description="Type of plot: 'bar' or 'pie'")
    data: Dict[str, float] = Field(..., description="Data for visualization")
    title: str = Field(..., description="Chart title")
    xlabel: Optional[str] = Field("", description="X-axis label (for bar charts)")
    ylabel: Optional[str] = Field("", description="Y-axis label (for bar charts)")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    k: int = Field(5, description="Number of results", ge=1, le=20)

class ImageClassificationResponse(BaseModel):
    status: str
    predicted_species: Optional[str] = None
    confidence: Optional[float] = None
    top3_predictions: Optional[List[Dict[str, Any]]] = None
    message: Optional[str] = None

# ============================================================
# 🏥 HEALTH CHECK
# ============================================================
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API status"""
    return {
        "service": "MeenaSetu AI API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    stats = meenasetu_ai.get_statistics()
    return {
        "status": "healthy",
        "ai_initialized": True,
        "ml_model_status": stats["ml_model_status"],
        "vector_db_documents": stats["database_stats"]["total_documents"],
        "timestamp": datetime.now().isoformat()
    }

# ============================================================
# 💬 RAG QUERY ENDPOINTS
# ============================================================
@app.post("/rag/query", response_model=QueryResponse, tags=["RAG"])
async def rag_query(req: QueryRequest):
    """
    Ask a question to MeenaSetu AI
    
    - **query**: Your question about fish, aquaculture, or related topics
    - **include_sources**: Whether to include source documents
    
    Returns intelligent answer with source attribution
    """
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    try:
        result = meenasetu_ai.ask(req.query, include_sources=req.include_sources)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"❌ Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/search", tags=["RAG"])
async def search_documents(req: SearchRequest):
    """
    Direct search in vector database
    
    - **query**: Search query
    - **k**: Number of results (1-20)
    
    Returns matching documents with metadata
    """
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    try:
        results = meenasetu_ai.search_documents(req.query, req.k)
        return {
            "query": req.query,
            "results_count": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# 📤 FILE UPLOAD ENDPOINTS
# ============================================================
@app.post("/upload/file", response_model=FileUploadResponse, tags=["Upload"])
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a single file (PDF, CSV, JSON, TXT, Image)
    
    - **file**: File to upload
    
    Supported formats: .pdf, .csv, .json, .txt, .jpg, .jpeg, .png, .gif, .bmp
    """
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    # Validate file type
    allowed_extensions = {'.pdf', '.csv', '.json', '.txt', '.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save file
    file_path = Config.UPLOADS_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"📥 File saved: {file.filename}")
        
        # Process file
        result = meenasetu_ai.upload_file(str(file_path))
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result.get("message"))
        
        return FileUploadResponse(**result)
    
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        if file_path.exists():
            file_path.unlink()  # Clean up on error
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/multiple", response_model=MultipleFilesUploadResponse, tags=["Upload"])
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """
    Upload multiple files at once
    
    - **files**: List of files to upload
    
    Returns summary of successful and failed uploads
    """
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    saved_files = []
    
    try:
        # Save all files first
        for file in files:
            file_path = Config.UPLOADS_DIR / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(str(file_path))
        
        # Process all files
        result = meenasetu_ai.upload_multiple_files(saved_files)
        return MultipleFilesUploadResponse(**result)
    
    except Exception as e:
        logger.error(f"❌ Multiple upload error: {e}")
        # Clean up saved files on error
        for file_path in saved_files:
            Path(file_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# 🖼️ IMAGE CLASSIFICATION ENDPOINTS
# ============================================================
@app.post("/classify/image", response_model=ImageClassificationResponse, tags=["Classification"])
async def classify_image(file: UploadFile = File(...)):
    """
    Classify fish species from image using ML model
    
    - **file**: Image file (jpg, jpeg, png, gif, bmp)
    
    Returns predicted species with confidence scores
    """
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    if not meenasetu_ai.fish_classifier.is_loaded:
        raise HTTPException(
            status_code=503, 
            detail="ML model not available. Please ensure model is trained and available."
        )
    
    # Validate image file
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in image_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image format: {file_ext}. Allowed: {', '.join(image_extensions)}"
        )
    
    # Save image temporarily
    temp_path = Config.UPLOADS_DIR / f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Classify
        result = meenasetu_ai.classify_image(str(temp_path))
        
        # Clean up
        temp_path.unlink()
        
        return ImageClassificationResponse(**result)
    
    except Exception as e:
        logger.error(f"❌ Classification error: {e}")
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# 📊 VISUALIZATION ENDPOINTS
# ============================================================
@app.post("/visualize/generate", tags=["Visualization"])
async def generate_visualization(req: VisualizationRequest):
    """
    Generate visualization from data
    
    - **plot_type**: 'bar' or 'pie'
    - **data**: Dictionary of labels to values
    - **title**: Chart title
    - **xlabel**: X-axis label (optional, for bar charts)
    - **ylabel**: Y-axis label (optional, for bar charts)
    
    Returns path to generated visualization
    """
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    try:
        result = meenasetu_ai.generate_visualization(
            req.plot_type, 
            req.data, 
            req.title,
            req.xlabel,
            req.ylabel
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result.get("message"))
        
        return result
    
    except Exception as e:
        logger.error(f"❌ Visualization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/visualize/analyze-csv", tags=["Visualization"])
async def analyze_csv_for_viz(file: UploadFile = File(...)):
    """
    Analyze CSV and suggest visualizations
    
    - **file**: CSV file to analyze
    
    Returns suggested visualizations with data
    """
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    # Save CSV temporarily
    csv_path = Config.UPLOADS_DIR / file.filename
    try:
        with open(csv_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Analyze
        result = meenasetu_ai.analyze_csv_for_visualization(str(csv_path))
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result.get("message"))
        
        return result
    
    except Exception as e:
        logger.error(f"❌ CSV analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualize/download/{filename}", tags=["Visualization"])
async def download_visualization(filename: str):
    """
    Download generated visualization
    
    - **filename**: Name of the visualization file
    
    Returns the image file
    """
    file_path = Config.OUTPUTS_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Visualization not found")
    
    return FileResponse(
        file_path,
        media_type="image/png",
        filename=filename
    )

# ============================================================
# 📈 STATISTICS & MONITORING ENDPOINTS
# ============================================================
@app.get("/stats", tags=["Statistics"])
async def get_statistics():
    """
    Get comprehensive system statistics
    
    Returns statistics about documents, queries, ML model, etc.
    """
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    stats = meenasetu_ai.get_statistics()
    return {
        "statistics": stats,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/conversation/history", tags=["Conversation"])
async def get_conversation_history():
    """
    Get conversation history
    
    Returns list of all user queries and AI responses
    """
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    history = meenasetu_ai.get_conversation_history()
    return {
        "conversation_count": len(history),
        "history": history,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/conversation/clear", tags=["Conversation"])
async def clear_conversation():
    """
    Clear conversation history
    
    Resets the conversation context
    """
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    meenasetu_ai.clear_conversation()
    return {
        "status": "success",
        "message": "Conversation history cleared",
        "timestamp": datetime.now().isoformat()
    }

# ============================================================
# 🔍 UTILITY ENDPOINTS
# ============================================================
@app.get("/files/list", tags=["Utilities"])
async def list_uploaded_files():
    """
    List all uploaded files
    
    Returns list of files in uploads directory
    """
    try:
        files = []
        for file_path in Config.UPLOADS_DIR.iterdir():
            if file_path.is_file():
                files.append({
                    "filename": file_path.name,
                    "size_bytes": file_path.stat().st_size,
                    "extension": file_path.suffix,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
        
        return {
            "total_files": len(files),
            "files": sorted(files, key=lambda x: x["modified"], reverse=True),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ List files error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config", tags=["Utilities"])
async def get_configuration():
    """
    Get system configuration
    
    Returns configuration paths and settings
    """
    return {
        "base_dir": str(Config.BASE_DIR),
        "vector_db_dir": str(Config.VECTOR_DB_DIR),
        "uploads_dir": str(Config.UPLOADS_DIR),
        "outputs_dir": str(Config.OUTPUTS_DIR),
        "embed_model": Config.EMBED_MODEL,
        "groq_model": Config.GROQ_MODEL,
        "chunk_size": Config.CHUNK_SIZE,
        "chunk_overlap": Config.CHUNK_OVERLAP,
        "retrieval_k": Config.RETRIEVAL_K,
        "device": str(Config.DEVICE),
        "timestamp": datetime.now().isoformat()
    }

# ============================================================
# 🧪 TEST ENDPOINT
# ============================================================
@app.get("/test", tags=["Testing"])
async def test_endpoint():
    """
    Test endpoint to verify API is working
    
    Returns test data and AI status
    """
    if meenasetu_ai is None:
        return {
            "status": "error",
            "message": "AI not initialized",
            "timestamp": datetime.now().isoformat()
        }
    
    return {
        "status": "success",
        "message": "MeenaSetu AI API is working correctly",
        "ai_initialized": True,
        "ml_model_loaded": meenasetu_ai.fish_classifier.is_loaded,
        "vector_db_docs": meenasetu_ai.vector_db.document_count,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================
# 🚀 RUN SERVER
# ============================================================
if __name__ == "__main__":
    import uvicorn
    
    # Determine the correct module path based on execution context
    script_path = Path(__file__).resolve()
    
    # If running from backend/app/main.py
    if script_path.parent.name == "app":
        module_path = "app.main:app"
    else:
        module_path = "main:app"
    
    print(f"🚀 Starting MeenaSetu AI API Server...")
    print(f"📍 Module: {module_path}")
    print(f"🌐 API Docs: http://localhost:8000/docs")
    print(f"📚 ReDoc: http://localhost:8000/redoc")
    
    uvicorn.run(
        module_path,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )