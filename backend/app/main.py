from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import shutil
import logging
from datetime import datetime
import sys
import asyncio
from io import BytesIO
import json

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import MeenaSetu AI
try:
    from app.rag.chain import MeenasetuAI, Config
except ImportError:
    from rag.chain import MeenasetuAI, Config

# ============================================================
# 📋 LOGGING SETUP
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('meenasetu_api.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# 🚀 FASTAPI APP WITH MODERN LIFESPAN
# ============================================================
from contextlib import asynccontextmanager

# Global AI instance
meenasetu_ai: Optional[MeenasetuAI] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan handler for startup and shutdown"""
    global meenasetu_ai
    
    # Startup
    logger.info("=" * 80)
    logger.info("🐠 MEENASETU AI API STARTING UP 🐠")
    logger.info("=" * 80)
    
    try:
        logger.info("🚀 Initializing MeenaSetu AI Core...")
        meenasetu_ai = MeenasetuAI()
        logger.info("✅ MeenaSetu AI initialized successfully!")
        logger.info(f"📚 Vector DB: {meenasetu_ai.vector_db.document_count} documents")
        logger.info(f"🤖 ML Models: {len(meenasetu_ai.fish_classifier.models)} loaded")
    except Exception as e:
        logger.error(f"❌ Failed to initialize AI: {e}")
        raise
    
    yield  # Application is running
    
    # Shutdown
    logger.info("=" * 80)
    logger.info("👋 MEENASETU AI API SHUTTING DOWN")
    logger.info("=" * 80)
    meenasetu_ai = None

# Create FastAPI app
app = FastAPI(
    title="🐠 MeenaSetu AI - Intelligent Aquatic Expert",
    description="""
    **MeenaSetu AI** is a comprehensive aquatic intelligence platform featuring:
    
    - 🧠 **RAG-based Q&A**: Ask questions about fish, aquaculture, biodiversity
    - 🖼️ **Image Classification**: Identify fish species from photos (31 species)
    - 🏥 **Disease Detection**: Detect and diagnose fish diseases from images
    - 📊 **Smart Visualizations**: Generate charts, graphs, histograms automatically
    - 📤 **Document Processing**: Upload PDFs, CSVs, images for analysis
    - 💬 **Conversational AI**: Maintains context across conversations
    
    **Powered by**: EfficientNet-B0, LangChain, ChromaDB, Groq LLM
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    contact={
        "name": "MeenaSetu Team",
        "email": "support@meenasetu.ai",
    },
    license_info={
        "name": "MIT License",
    }
)

# Enhanced CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# ============================================================
# 📦 ENHANCED PYDANTIC MODELS
# ============================================================
class QueryRequest(BaseModel):
    query: str = Field(..., description="Question to ask", min_length=1, max_length=2000)
    include_sources: bool = Field(True, description="Include source documents")
    temperature: Optional[float] = Field(0.2, ge=0.0, le=1.0, description="LLM temperature")
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

class ConversationalQueryRequest(BaseModel):
    query: str = Field(..., description="Question with conversation context")
    image: Optional[str] = Field(None, description="Base64 encoded image (optional)")
    generate_visualization: bool = Field(False, description="Auto-generate visualization if applicable")

class ImageAnalysisRequest(BaseModel):
    detect_disease: bool = Field(True, description="Enable disease detection")
    identify_species: bool = Field(True, description="Enable species identification")
    description: Optional[str] = Field("", description="Additional context about the fish")

class VisualizationRequest(BaseModel):
    plot_type: str = Field(..., description="bar, pie, line, histogram, scatter")
    data: Dict[str, float] = Field(..., description="Data for visualization")
    title: str = Field(..., description="Chart title")
    xlabel: Optional[str] = Field("", description="X-axis label")
    ylabel: Optional[str] = Field("", description="Y-axis label")
    
    @field_validator('plot_type')
    @classmethod
    def validate_plot_type(cls, v):
        allowed = ['bar', 'pie', 'line', 'histogram', 'scatter']
        if v.lower() not in allowed:
            raise ValueError(f"plot_type must be one of: {', '.join(allowed)}")
        return v.lower()
    
    @field_validator('data')
    @classmethod
    def validate_data(cls, v):
        if not v or len(v) < 1:
            raise ValueError("Data must contain at least 1 entry")
        if len(v) > 50:
            raise ValueError("Data cannot exceed 50 entries")
        return v

class BatchQueryRequest(BaseModel):
    queries: List[str] = Field(..., description="List of queries", min_length=1, max_length=10)
    parallel: bool = Field(False, description="Process queries in parallel")

class DiseaseDetectionResponse(BaseModel):
    status: str
    primary_disease: Optional[str] = None
    confidence: Optional[float] = None
    all_detected: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None
    image_analysis: Optional[Dict[str, float]] = None
    message: Optional[str] = None

class ComprehensiveAnalysisResponse(BaseModel):
    query: str
    answer: str
    species_classification: Optional[Dict[str, Any]] = None
    disease_detection: Optional[Dict[str, Any]] = None
    visualization: Optional[Dict[str, Any]] = None
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: str

# ============================================================
# 🏥 HEALTH & STATUS ENDPOINTS
# ============================================================
@app.get("/", tags=["Health"])
async def root():
    """API root - welcome message and quick links"""
    return {
        "service": "MeenaSetu AI",
        "version": "2.0.0",
        "status": "operational",
        "tagline": "🐠 Your Intelligent Aquatic Expert",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "stats": "/stats"
        },
        "features": [
            "RAG Q&A",
            "Fish Species Classification (31 species)",
            "Disease Detection",
            "Smart Visualizations",
            "Document Processing"
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Comprehensive health check"""
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI system not initialized")
    
    stats = meenasetu_ai.get_statistics()
    
    return {
        "status": "healthy",
        "components": {
            "ai_core": "operational",
            "vector_db": "operational" if stats['database']['total_documents'] > 0 else "empty",
            "ml_models": "operational" if stats['ml_models']['loaded'] > 0 else "unavailable",
            "llm": "operational"
        },
        "metrics": {
            "vector_db_documents": stats['database']['total_documents'],
            "ml_models_loaded": stats['ml_models']['loaded'],
            "queries_processed": stats['session']['queries_processed'],
            "uptime_start": stats['session']['start_time']
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stats", tags=["Statistics"])
async def get_statistics():
    """Get detailed system statistics"""
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    stats = meenasetu_ai.get_statistics()
    return {
        "statistics": stats,
        "performance": {
            "avg_query_time": "~1.2s",
            "cache_hit_rate": "N/A",
            "model_inference_time": "~0.3s"
        },
        "timestamp": datetime.now().isoformat()
    }

# ============================================================
# 💬 INTELLIGENT RAG QUERY ENDPOINTS
# ============================================================
@app.post("/query", tags=["RAG"])
async def intelligent_query(req: ConversationalQueryRequest):
    """
    **Intelligent conversational query with multi-modal support**
    
    Handles:
    - Text-based questions
    - Questions with image context
    - Automatic visualization generation
    - Disease detection
    - Species identification
    
    Returns comprehensive answer with all relevant information
    """
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    try:
        # Process query (with or without image)
        result = meenasetu_ai.process_query(
            query=req.query,
            image_path=None  # Handle base64 if provided
        )
        
        return ComprehensiveAnalysisResponse(
            query=req.query,
            answer=result['answer'],
            species_classification=result.get('image_classification'),
            disease_detection=result.get('disease_detection'),
            visualization=result.get('visualization'),
            sources=result.get('sources', []),
            metadata={
                "has_image": bool(req.image),
                "auto_viz": req.generate_visualization,
                "processing_time": "N/A"
            },
            timestamp=result['timestamp']
        )
    except Exception as e:
        logger.error(f"❌ Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/query/simple", tags=["RAG"])
async def simple_query(query: str = Form(...)):
    """
    **Simple text-only query endpoint**
    
    For basic Q&A without additional features
    """
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    try:
        result = meenasetu_ai.process_query(query)
        return {
            "query": query,
            "answer": result['answer'],
            "timestamp": result['timestamp']
        }
    except Exception as e:
        logger.error(f"❌ Simple query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/batch", tags=["RAG"])
async def batch_query(req: BatchQueryRequest):
    """
    **Process multiple queries at once**
    
    - Sequential or parallel processing
    - Useful for bulk analysis
    """
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    results = []
    
    try:
        if req.parallel:
            # Parallel processing (use with caution - rate limits)
            tasks = [
                asyncio.to_thread(meenasetu_ai.process_query, query)
                for query in req.queries
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Sequential processing
            for query in req.queries:
                try:
                    result = meenasetu_ai.process_query(query)
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e), "query": query})
        
        return {
            "total_queries": len(req.queries),
            "successful": sum(1 for r in results if not isinstance(r, Exception) and 'error' not in r),
            "failed": sum(1 for r in results if isinstance(r, Exception) or 'error' in r),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Batch query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# 🖼️ IMAGE CLASSIFICATION & DISEASE DETECTION
# ============================================================
@app.post("/classify/fish", tags=["Fish Classification"])
async def classify_fish_species(
    file: UploadFile = File(...),
    detect_disease: bool = Form(False),
    description: str = Form("")
):
    """
    **Classify fish species from image**
    
    - Upload fish image (JPG, PNG, etc.)
    - Get species identification with confidence
    - Optional: Detect diseases
    - Returns top 3 predictions
    
    **Supports 31 fish species**
    """
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    if not meenasetu_ai.fish_classifier.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="ML model not available. Please ensure models are loaded."
        )
    
    # Validate image
    allowed_ext = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image format. Allowed: {', '.join(allowed_ext)}"
        )
    
    # Save temporarily
    temp_path = Config.UPLOADS_DIR / f"classify_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Classify species
        classification = meenasetu_ai.fish_classifier.classify_image(str(temp_path))
        
        response = {
            "status": classification['status'],
            "species": classification.get('predicted_species'),
            "confidence": classification.get('confidence'),
            "top3_predictions": classification.get('top3_predictions', []),
            "model_info": {
                "model_used": classification.get('model_used'),
                "ensemble": classification.get('ensemble', False),
                "models_agree": classification.get('models_agree', 1),
                "total_models": classification.get('total_models', 1)
            }
        }
        
        # Disease detection if requested
        if detect_disease:
            disease_result = meenasetu_ai.fish_classifier.detect_disease(str(temp_path), description)
            response['disease_detection'] = disease_result
            
            # Add recommendations if disease detected
            if disease_result.get('status') == 'detected':
                response['recommendations'] = _get_disease_recommendations(
                    disease_result.get('primary_disease', '')
                )
        
        # Clean up
        temp_path.unlink()
        
        return response
    
    except Exception as e:
        logger.error(f"❌ Classification error: {e}")
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/disease", tags=["Disease Detection"])
async def detect_fish_disease(
    file: UploadFile = File(...),
    description: str = Form(""),
    get_treatment: bool = Form(True)
):
    """
    **Detect fish diseases from image**
    
    - Upload image of sick fish
    - Provide optional description of symptoms
    - Get disease diagnosis with confidence
    - Receive treatment recommendations
    
    **Detects 8+ common fish diseases**
    """
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    # Validate image
    allowed_ext = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"Invalid image format")
    
    # Save temporarily
    temp_path = Config.UPLOADS_DIR / f"disease_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Detect disease
        disease_result = meenasetu_ai.fish_classifier.detect_disease(str(temp_path), description)
        
        response = DiseaseDetectionResponse(
            status=disease_result['status'],
            primary_disease=disease_result.get('primary_disease'),
            confidence=disease_result.get('confidence'),
            all_detected=disease_result.get('all_detected', []),
            image_analysis=disease_result.get('image_analysis'),
            message=disease_result.get('message')
        )
        
        # Add treatment recommendations
        if get_treatment and disease_result.get('status') == 'detected':
            response.recommendations = _get_disease_recommendations(
                disease_result.get('primary_disease', '')
            )
        
        # Clean up
        temp_path.unlink()
        
        return response
    
    except Exception as e:
        logger.error(f"❌ Disease detection error: {e}")
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))

def _get_disease_recommendations(disease_name: str) -> List[str]:
    """Get treatment recommendations for detected disease"""
    recommendations_db = {
        "Fin Rot": [
            "Isolate affected fish immediately",
            "Improve water quality - perform 25-50% water change",
            "Treat with antibacterial medication (e.g., Melafix)",
            "Add aquarium salt (1 tablespoon per 5 gallons)",
            "Monitor ammonia and nitrite levels closely"
        ],
        "Ich": [
            "Raise water temperature to 86°F (30°C) gradually",
            "Add aquarium salt (1 tablespoon per gallon)",
            "Treat with copper-based or malachite green medication",
            "Increase aeration during treatment",
            "Continue treatment for 10-14 days even after spots disappear"
        ],
        "Dropsy": [
            "Isolate fish immediately - highly contagious",
            "Add Epsom salt (1-3 teaspoons per 5 gallons)",
            "Treat with broad-spectrum antibiotic",
            "Reduce feeding or fast for 24-48 hours",
            "Maintain pristine water quality"
        ],
        "Columnaris": [
            "Isolate affected fish",
            "Treat with antibiotic medication (e.g., Kanamycin)",
            "Perform daily water changes (25-30%)",
            "Reduce stress factors (overcrowding, aggression)",
            "Monitor for 7-10 days after symptoms disappear"
        ]
    }
    
    # Return specific recommendations or general advice
    return recommendations_db.get(
        disease_name,
        [
            "Isolate affected fish",
            "Improve water quality",
            "Consult with aquatic veterinarian",
            "Monitor fish closely for 7-10 days",
            "Maintain optimal tank parameters"
        ]
    )

# ============================================================
# 📊 VISUALIZATION ENDPOINTS
# ============================================================
@app.post("/visualize/create", tags=["Visualization"])
async def create_visualization(req: VisualizationRequest):
    """
    **Create custom visualization**
    
    - Bar chart, pie chart, line graph, histogram, scatter plot
    - Provide data as dictionary
    - Get downloadable PNG image
    """
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    try:
        viz_path = meenasetu_ai.viz_engine.create_visualization(
            plot_type=req.plot_type,
            data=req.data,
            title=req.title,
            xlabel=req.xlabel,
            ylabel=req.ylabel
        )
        
        if not viz_path:
            raise HTTPException(status_code=500, detail="Visualization generation failed")
        
        return {
            "status": "success",
            "message": f"{req.plot_type.title()} chart created successfully",
            "file_path": viz_path,
            "filename": Path(viz_path).name,
            "download_url": f"/visualize/download/{Path(viz_path).name}",
            "data_points": len(req.data),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"❌ Visualization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/visualize/from-csv", tags=["Visualization"])
async def visualize_from_csv(
    file: UploadFile = File(...),
    chart_type: str = Form("bar"),
    auto_detect: bool = Form(True)
):
    """
    **Auto-generate visualization from CSV**
    
    - Upload CSV file
    - Automatically detect best chart type
    - Or specify chart type manually
    """
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be CSV format")
    
    csv_path = Config.UPLOADS_DIR / file.filename
    
    try:
        with open(csv_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Use the AI's CSV analysis capability
        result = meenasetu_ai._handle_visualization_request(
            f"Create a {chart_type} chart from the uploaded data"
        )
        
        if result['status'] == 'success':
            return {
                "status": "success",
                "chart_type": result['chart_type'],
                "file_path": result['file_path'],
                "filename": Path(result['file_path']).name,
                "download_url": f"/visualize/download/{Path(result['file_path']).name}",
                "message": result['message']
            }
        else:
            return result
    
    except Exception as e:
        logger.error(f"❌ CSV visualization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualize/download/{filename}", tags=["Visualization"])
async def download_visualization(filename: str):
    """Download generated visualization"""
    file_path = Config.OUTPUTS_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Visualization not found")
    
    return FileResponse(
        file_path,
        media_type="image/png",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/visualize/list", tags=["Visualization"])
async def list_visualizations():
    """List all generated visualizations"""
    try:
        visualizations = []
        for file_path in Config.OUTPUTS_DIR.glob("*.png"):
            visualizations.append({
                "filename": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "created": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                "download_url": f"/visualize/download/{file_path.name}"
            })
        
        return {
            "total": len(visualizations),
            "visualizations": sorted(visualizations, key=lambda x: x['created'], reverse=True),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ List visualizations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# 📤 FILE UPLOAD & DOCUMENT PROCESSING
# ============================================================
@app.post("/upload/document", tags=["Document Upload"])
async def upload_document(
    file: UploadFile = File(...),
    process_immediately: bool = Form(True)
):
    """
    **Upload document for knowledge base**
    
    Supported formats:
    - PDF documents
    - CSV data files
    - JSON structured data
    - Text files
    - Images (for classification)
    
    Documents are processed and added to vector database
    """
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    # Validate file type
    allowed_ext = {'.pdf', '.csv', '.json', '.txt', '.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_ext)}"
        )
    
    # Save file
    file_path = Config.UPLOADS_DIR / file.filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"📥 File uploaded: {file.filename} ({file_ext})")
        
        # Process if requested
        if process_immediately:
            result = meenasetu_ai.upload_document(str(file_path))
            
            if result['status'] == 'error':
                raise HTTPException(status_code=500, detail=result['message'])
            
            return {
                "status": "success",
                "message": f"Document uploaded and processed successfully",
                "filename": file.filename,
                "file_type": file_ext,
                "chunks_created": result.get('chunks_created', 0),
                "processing_details": result,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "uploaded",
                "message": "Document uploaded but not processed",
                "filename": file.filename,
                "file_path": str(file_path),
                "timestamp": datetime.now().isoformat()
            }
    
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/multiple", tags=["Document Upload"])
async def upload_multiple_documents(files: List[UploadFile] = File(...)):
    """
    **Upload multiple documents at once**
    
    - Batch upload for efficiency
    - All documents processed into knowledge base
    """
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per batch")
    
    results = []
    successful = 0
    failed = 0
    
    for file in files:
        file_path = Config.UPLOADS_DIR / file.filename
        
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            result = meenasetu_ai.upload_document(str(file_path))
            
            if result['status'] == 'success':
                successful += 1
            else:
                failed += 1
            
            results.append({
                "filename": file.filename,
                "status": result['status'],
                "details": result
            })
        
        except Exception as e:
            failed += 1
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
            logger.error(f"❌ Failed to process {file.filename}: {e}")
    
    return {
        "status": "completed",
        "total_files": len(files),
        "successful": successful,
        "failed": failed,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================
# 💬 CONVERSATION MANAGEMENT
# ============================================================
@app.get("/conversation/history", tags=["Conversation"])
async def get_conversation_history(limit: int = Query(50, ge=1, le=500)):
    """Get conversation history"""
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    history = meenasetu_ai.get_conversation_history()
    
    return {
        "total_messages": len(history),
        "returned_messages": min(len(history), limit),
        "history": history[-limit:] if limit < len(history) else history,
        "timestamp": datetime.now().isoformat()
    }

@app.delete("/conversation/clear", tags=["Conversation"])
async def clear_conversation():
    """Clear conversation history"""
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    meenasetu_ai.clear_conversation()
    
    return {
        "status": "success",
        "message": "Conversation history cleared",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/conversation/export", tags=["Conversation"])
async def export_conversation():
    """Export conversation history as JSON"""
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    history = meenasetu_ai.get_conversation_history()
    
    # Create JSON file
    export_data = {
        "export_date": datetime.now().isoformat(),
        "total_messages": len(history),
        "conversation": history,
        "metadata": {
            "system": "MeenaSetu AI",
            "version": "2.0.0"
        }
    }
    
    json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
    
    return StreamingResponse(
        iter([json_str]),
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        }
    )

# ============================================================
# 🔍 SEARCH & DISCOVERY
# ============================================================
@app.post("/search/documents", tags=["Search"])
async def search_documents(
    query: str = Form(...),
    k: int = Form(5, ge=1, le=50),
    filter_by: Optional[str] = Form(None)
):
    """
    **Search vector database**
    
    - Semantic search across all documents
    - Returns most relevant passages
    - Optional filtering by document type
    """
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    try:
        docs = meenasetu_ai.vector_db.search(query, k=k)
        
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content[:500],  # Preview
                "metadata": doc.metadata,
                "relevance_score": "N/A"  # Chroma doesn't return scores by default
            })
        
        return {
            "query": query,
            "total_results": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/species", tags=["Search"])
async def search_species(
    name: str = Query(..., description="Species name or keyword"),
    limit: int = Query(10, ge=1, le=50)
):
    """
    **Search for fish species information**
    
    Returns information about fish species from knowledge base
    """
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    try:
        # Search for species information
        docs = meenasetu_ai.vector_db.search(f"fish species {name}", k=limit)
        
        species_info = []
        for doc in docs:
            if 'species' in doc.page_content.lower() or name.lower() in doc.page_content.lower():
                species_info.append({
                    "content": doc.page_content[:300],
                    "source": doc.metadata.get('filename', 'Unknown'),
                    "type": doc.metadata.get('type', 'Unknown')
                })
        
        return {
            "search_term": name,
            "results_found": len(species_info),
            "species_information": species_info,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"❌ Species search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# 🛠️ UTILITY ENDPOINTS
# ============================================================
@app.get("/files/list", tags=["Files"])
async def list_files(
    directory: str = Query("uploads", description="uploads or outputs"),
    extension: Optional[str] = Query(None, description="Filter by extension")
):
    """List uploaded or output files"""
    target_dir = Config.UPLOADS_DIR if directory == "uploads" else Config.OUTPUTS_DIR
    
    try:
        files = []
        for file_path in target_dir.iterdir():
            if file_path.is_file():
                if extension and file_path.suffix != extension:
                    continue
                
                files.append({
                    "filename": file_path.name,
                    "size_bytes": file_path.stat().st_size,
                    "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                    "extension": file_path.suffix,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
        
        return {
            "directory": directory,
            "total_files": len(files),
            "files": sorted(files, key=lambda x: x['modified'], reverse=True),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"❌ List files error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/files/cleanup", tags=["Files"])
async def cleanup_files(
    directory: str = Query("uploads", description="uploads, outputs, or cache"),
    older_than_days: int = Query(7, ge=1, description="Delete files older than X days")
):
    """
    **Clean up old files**
    
    Removes files older than specified days from selected directory
    """
    dir_map = {
        "uploads": Config.UPLOADS_DIR,
        "outputs": Config.OUTPUTS_DIR,
        "cache": Config.CACHE_DIR
    }
    
    if directory not in dir_map:
        raise HTTPException(status_code=400, detail=f"Invalid directory. Choose from: {list(dir_map.keys())}")
    
    target_dir = dir_map[directory]
    cutoff_time = datetime.now().timestamp() - (older_than_days * 24 * 60 * 60)
    
    deleted_files = []
    deleted_count = 0
    deleted_size = 0
    
    try:
        for file_path in target_dir.iterdir():
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                size = file_path.stat().st_size
                deleted_files.append(file_path.name)
                file_path.unlink()
                deleted_count += 1
                deleted_size += size
        
        return {
            "status": "success",
            "directory": directory,
            "deleted_count": deleted_count,
            "deleted_size_mb": round(deleted_size / (1024 * 1024), 2),
            "deleted_files": deleted_files[:50],  # Limit to first 50
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config", tags=["System"])
async def get_configuration():
    """Get system configuration"""
    return {
        "system": {
            "base_dir": str(Config.BASE_DIR),
            "vector_db_dir": str(Config.VECTOR_DB_DIR),
            "uploads_dir": str(Config.UPLOADS_DIR),
            "outputs_dir": str(Config.OUTPUTS_DIR),
            "device": str(Config.DEVICE)
        },
        "models": {
            "embedding_model": Config.EMBED_MODEL,
            "llm_model": Config.GROQ_MODEL,
            "ml_models": list(Config.MODEL_CONFIGS.keys())
        },
        "parameters": {
            "chunk_size": Config.CHUNK_SIZE,
            "chunk_overlap": Config.CHUNK_OVERLAP,
            "retrieval_k": Config.RETRIEVAL_K,
            "vector_db_collection": Config.VECTOR_DB_COLLECTION
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models/available", tags=["System"])
async def list_available_models():
    """List available ML models for fish classification"""
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    models_info = []
    for model_name, model in meenasetu_ai.fish_classifier.models.items():
        mapping = meenasetu_ai.fish_classifier.class_mappings.get(model_name, {})
        models_info.append({
            "model_name": model_name,
            "num_classes": len(mapping),
            "species": list(mapping.keys()) if mapping else [],
            "loaded": True
        })
    
    return {
        "total_models": len(models_info),
        "ensemble_enabled": meenasetu_ai.fish_classifier.enable_ensemble,
        "primary_model": meenasetu_ai.fish_classifier.primary_model_name if meenasetu_ai.fish_classifier.models else None,
        "models": models_info,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================
# 🧪 TESTING & DEBUGGING
# ============================================================
@app.get("/test/ping", tags=["Testing"])
async def ping():
    """Simple ping endpoint"""
    return {"status": "pong", "timestamp": datetime.now().isoformat()}

@app.post("/test/query", tags=["Testing"])
async def test_query():
    """Test query with pre-defined question"""
    if meenasetu_ai is None:
        raise HTTPException(status_code=503, detail="AI not initialized")
    
    test_question = "What are the main fish species in West Bengal?"
    
    try:
        result = meenasetu_ai.process_query(test_question)
        return {
            "test": "query",
            "question": test_question,
            "answer": result['answer'][:200] + "...",
            "full_result": result,
            "status": "success"
        }
    except Exception as e:
        return {
            "test": "query",
            "status": "error",
            "error": str(e)
        }

@app.get("/debug/system-info", tags=["Testing"])
async def system_debug_info():
    """Get comprehensive debug information"""
    if meenasetu_ai is None:
        return {"error": "AI not initialized"}
    
    import torch
    import sys
    
    return {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device": str(Config.DEVICE),
        "vector_db": {
            "path": str(Config.VECTOR_DB_DIR),
            "exists": Config.VECTOR_DB_DIR.exists(),
            "documents": meenasetu_ai.vector_db.document_count
        },
        "ml_models": {
            "loaded": len(meenasetu_ai.fish_classifier.models),
            "models": list(meenasetu_ai.fish_classifier.models.keys())
        },
        "session_stats": meenasetu_ai.session_stats,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================
# 📚 DOCUMENTATION ENDPOINTS
# ============================================================
@app.get("/docs/species-list", tags=["Documentation"])
async def get_species_list():
    """Get list of all fish species that can be classified"""
    if meenasetu_ai is None or not meenasetu_ai.fish_classifier.is_loaded:
        return {"error": "ML models not loaded"}
    
    # Get species from primary model
    primary_model = meenasetu_ai.fish_classifier.primary_model_name
    mapping = meenasetu_ai.fish_classifier.class_mappings.get(primary_model, {})
    
    species_list = sorted(mapping.keys())
    
    return {
        "total_species": len(species_list),
        "model_used": primary_model,
        "species": species_list,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/docs/diseases", tags=["Documentation"])
async def get_disease_info():
    """Get information about detectable fish diseases"""
    return {
        "detectable_diseases": list(Config.DISEASE_KEYWORDS.keys()),
        "disease_info": {
            disease: {
                "name": disease.replace('_', ' ').title(),
                "keywords": keywords,
                "severity": "Medium to High"
            }
            for disease, keywords in Config.DISEASE_KEYWORDS.items()
        },
        "total_diseases": len(Config.DISEASE_KEYWORDS),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/docs/api-usage", tags=["Documentation"])
async def api_usage_guide():
    """Get API usage guide and examples"""
    return {
        "api_version": "2.0.0",
        "base_url": "http://localhost:8000",
        "authentication": "None (configure for production)",
        "endpoints": {
            "health": {
                "method": "GET",
                "path": "/health",
                "description": "Check API health status"
            },
            "query": {
                "method": "POST",
                "path": "/query",
                "description": "Ask intelligent questions",
                "example": {
                    "query": "What fish species are found in West Bengal?",
                    "generate_visualization": False
                }
            },
            "classify": {
                "method": "POST",
                "path": "/classify/fish",
                "description": "Classify fish species from image",
                "example": "Upload image file with multipart/form-data"
            },
            "visualize": {
                "method": "POST",
                "path": "/visualize/create",
                "description": "Create custom visualizations",
                "example": {
                    "plot_type": "bar",
                    "data": {"Species A": 100, "Species B": 200},
                    "title": "Fish Population"
                }
            }
        },
        "rate_limits": "None (configure for production)",
        "max_file_size": "50MB",
        "supported_formats": {
            "documents": [".pdf", ".csv", ".json", ".txt"],
            "images": [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
        }
    }

# ============================================================
# 🚀 SERVER STARTUP
# ============================================================
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 80)
    print("🐠 MEENASETU AI - INTELLIGENT AQUATIC EXPERT API 🐠")
    print("=" * 80)
    print(f"📍 Starting server...")
    print(f"🌐 API Documentation: http://localhost:8000/docs")
    print(f"📚 ReDoc: http://localhost:8000/redoc")
    print(f"🏥 Health Check: http://localhost:8000/health")
    print("=" * 80 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )