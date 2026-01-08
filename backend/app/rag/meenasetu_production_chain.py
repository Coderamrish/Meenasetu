import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# PyTorch for Species Classification
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# TensorFlow/Keras for Disease Detection
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available - disease detection disabled")

# LangChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv

# ============================================================
# 📋 LOGGING & ENVIRONMENT
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('meenasetu_production.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment
env_paths = [
    Path(__file__).parent.parent.parent / ".env",
    Path(__file__).parent.parent / ".env",
    Path(__file__).parent / ".env",
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"✓ Loaded .env from: {env_path}")
        break

# ============================================================
# ⚙️ CONFIGURATION - FIXED PATH RESOLUTION
# ============================================================
class Config:
    """Production configuration with fixed paths"""
    
    # Get script directory
    SCRIPT_DIR = Path(__file__).resolve().parent
    
    # Method 1: Try to find BASE_DIR by looking for 'training' folder
    BASE_DIR = None
    current = SCRIPT_DIR
    for _ in range(6):  # Search up to 6 levels
        if (current / "training" / "checkpoints").exists():
            BASE_DIR = current
            logger.info(f"✓ Found BASE_DIR: {BASE_DIR}")
            break
        current = current.parent
    
    # Method 2: Fallback to hardcoded path if search fails
    if BASE_DIR is None:
        BASE_DIR = Path(r"C:\Users\AMRISH\Documents\Meenasetu")
        logger.warning(f"⚠️ Using hardcoded BASE_DIR: {BASE_DIR}")
    
    # Verify BASE_DIR is correct
    if not (BASE_DIR / "training" / "checkpoints").exists():
        logger.error(f"❌ CRITICAL: Cannot find training/checkpoints in {BASE_DIR}")
        logger.error(f"   Please check your project structure!")
    
    # Directories
    VECTOR_DB_DIR = BASE_DIR / "models" / "vector_db"
    UPLOADS_DIR = BASE_DIR / "backend" / "uploads"
    OUTPUTS_DIR = BASE_DIR / "backend" / "outputs"
    CACHE_DIR = BASE_DIR / "backend" / "cache"
    TRAINING_DIR = BASE_DIR / "training"
    CHECKPOINTS_DIR = TRAINING_DIR / "checkpoints"
    
    # Log paths for debugging
    logger.info(f"📁 Project Structure:")
    logger.info(f"   BASE_DIR: {BASE_DIR}")
    logger.info(f"   TRAINING_DIR: {TRAINING_DIR} (exists: {TRAINING_DIR.exists()})")
    logger.info(f"   CHECKPOINTS_DIR: {CHECKPOINTS_DIR} (exists: {CHECKPOINTS_DIR.exists()})")
    
    # Species Classification Models
    SPECIES_MODEL_CONFIGS = {
        'fish_species_model': {
            'path': CHECKPOINTS_DIR / "fish_model.pth",
            'class_mapping': CHECKPOINTS_DIR / "class_mapping1.json",
            'priority': 1
        },
        'best_model': {
            'path': CHECKPOINTS_DIR / "best_model.pth",
            'class_mapping': CHECKPOINTS_DIR / "class_mapping.json",
            'priority': 2
        },
        'final_model': {
            'path': CHECKPOINTS_DIR / "final_model.pth",
            'class_mapping': CHECKPOINTS_DIR / "class_mapping.json",
            'priority': 3
        }
    }
    
    # Disease Detection Models (Keras)
    DISEASE_MODEL_CONFIGS = {
        'disease_model_final': {
            'path': CHECKPOINTS_DIR / "final.keras",
            'class_mapping': CHECKPOINTS_DIR / "classes2.json",
            'priority': 1
        },
        'disease_model_s1': {
            'path': CHECKPOINTS_DIR / "s1.keras",
            'class_mapping': CHECKPOINTS_DIR / "classes2.json",
            'priority': 2
        }
    }
    
    # LLM Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    # Vector DB
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_DB_COLLECTION = "meenasetu_enhanced"
    RETRIEVAL_K = 10
    
    # Processing
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def setup_directories():
        """Create necessary directories"""
        for directory in [Config.UPLOADS_DIR, Config.OUTPUTS_DIR, Config.CACHE_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        logger.info("✓ Directories setup complete")

Config.setup_directories()

# ============================================================
# 🏥 FIXED DISEASE DETECTION SYSTEM
# ============================================================
class FishDiseaseDetector:
    """Fish disease detection - FIXED VERSION"""
    
    def __init__(self):
        self.models = {}
        self.class_mappings = {}
        self.primary_model_name = None
        self.is_available = TENSORFLOW_AVAILABLE
        
        if not self.is_available:
            logger.warning("🏥 Disease Detection DISABLED (TensorFlow not available)")
            return
        
        logger.info("🏥 Initializing Fish Disease Detection System")
        self._load_all_models()
    
    def _load_class_mapping(self, mapping_path: Path) -> Optional[Dict[str, int]]:
        """Load and normalize class mapping to disease_name -> id format"""
        try:
            with open(str(mapping_path), 'r', encoding='utf-8') as f:
                mapping_raw = json.load(f)
            
            logger.info(f"   📋 Raw mapping type: {type(mapping_raw)}")
            logger.info(f"   📋 Sample: {list(mapping_raw.items())[:3]}")
            
            # Normalize to disease_name -> id format
            disease_to_id = {}
            
            # Case 1: {"0": "disease", "1": "disease"} - id_to_disease format
            if all(str(k).isdigit() for k in mapping_raw.keys()):
                logger.info("   ✓ Format: id_to_disease (numeric string keys)")
                disease_to_id = {disease_name: int(id_str) 
                                for id_str, disease_name in mapping_raw.items()}
            
            # Case 2: {"disease_to_id": {...}}
            elif 'disease_to_id' in mapping_raw:
                logger.info("   ✓ Format: disease_to_id nested")
                disease_to_id = {k: int(v) for k, v in mapping_raw['disease_to_id'].items()}
            
            # Case 3: {"id_to_disease": {"0": "disease"}}
            elif 'id_to_disease' in mapping_raw:
                logger.info("   ✓ Format: id_to_disease nested")
                disease_to_id = {disease_name: int(id_str) 
                                for id_str, disease_name in mapping_raw['id_to_disease'].items()}
            
            # Case 4: {"disease": 0, "disease": 1} - already disease_to_id
            elif all(isinstance(v, (int, str)) and (isinstance(v, int) or str(v).isdigit()) 
                    for v in mapping_raw.values()):
                logger.info("   ✓ Format: direct disease_to_id")
                disease_to_id = {k: int(v) for k, v in mapping_raw.items()}
            
            else:
                logger.warning(f"   ✗ Unknown format. Keys: {list(mapping_raw.keys())[:3]}")
                return None
            
            if not disease_to_id:
                logger.error("   ✗ Mapping is empty after processing")
                return None
            
            logger.info(f"   ✓ Loaded {len(disease_to_id)} disease classes")
            logger.info(f"   📋 Classes: {list(disease_to_id.keys())[:5]}...")
            
            return disease_to_id
                
        except Exception as e:
            logger.error(f"   ✗ Error loading mapping: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _load_all_models(self):
        """Load disease models with robust error handling"""
        available_models = []
        
        for model_name, config in Config.DISEASE_MODEL_CONFIGS.items():
            model_path = config['path']
            mapping_path = config['class_mapping']
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 Attempting: {model_name}")
            logger.info(f"   Model: {model_path}")
            logger.info(f"   Exists: {model_path.exists()}")
            logger.info(f"   Mapping: {mapping_path}")
            logger.info(f"   Exists: {mapping_path.exists()}")
            
            if not model_path.exists():
                logger.warning(f"   ✗ Model file not found")
                continue
                
            if not mapping_path.exists():
                logger.warning(f"   ✗ Mapping file not found")
                continue
            
            try:
                # Step 1: Load class mapping FIRST
                logger.info(f"   📥 Step 1: Loading class mapping...")
                class_mapping = self._load_class_mapping(mapping_path)
                
                if not class_mapping or len(class_mapping) == 0:
                    logger.error(f"   ✗ Invalid or empty class mapping")
                    continue
                
                # Step 2: Load Keras model
                logger.info(f"   📥 Step 2: Loading Keras model...")
                model = keras.models.load_model(str(model_path), compile=False)
                logger.info(f"   ✓ Model loaded: {model.input_shape} -> {model.output_shape}")
                
                # Step 3: Verify dimensions
                model_classes = model.output_shape[-1]
                mapping_classes = len(class_mapping)
                logger.info(f"   🔍 Step 3: Verifying dimensions...")
                logger.info(f"      Model output: {model_classes} classes")
                logger.info(f"      Mapping has: {mapping_classes} classes")
                
                if model_classes != mapping_classes:
                    logger.error(f"   ✗ DIMENSION MISMATCH!")
                    logger.error(f"      Model expects {model_classes} but mapping has {mapping_classes}")
                    continue
                
                # Success!
                self.models[model_name] = model
                self.class_mappings[model_name] = class_mapping
                available_models.append((config['priority'], model_name))
                
                logger.info(f"   ✅ SUCCESS! {model_name} loaded")
                
            except Exception as e:
                logger.error(f"   ✗ Failed to load {model_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info(f"\n{'='*60}")
        if available_models:
            available_models.sort(key=lambda x: x[0])
            self.primary_model_name = available_models[0][1]
            logger.info(f"🎯 Primary disease model: {self.primary_model_name}")
            logger.info(f"✅ Disease detection ready ({len(self.models)} model(s))")
            
            # Log available diseases
            primary_mapping = self.class_mappings[self.primary_model_name]
            logger.info(f"📋 Available diseases ({len(primary_mapping)}):")
            for disease, idx in sorted(primary_mapping.items(), key=lambda x: x[1])[:10]:
                logger.info(f"   [{idx}] {disease}")
        else:
            logger.warning("⚠️ No disease models loaded")
    
    def detect_disease(self, image_path: str) -> Dict[str, Any]:
        """Detect fish disease from image"""
        if not self.is_available or not self.models:
            return {
                "status": "unavailable", 
                "message": "Disease detection not available"
            }
        
        try:
            # Preprocess image
            img = Image.open(image_path).convert('RGB').resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Get model and mapping
            model = self.models[self.primary_model_name]
            class_mapping = self.class_mappings[self.primary_model_name]
            id_to_disease = {v: k for k, v in class_mapping.items()}
            
            # Predict
            predictions = model.predict(img_array, verbose=0)
            predicted_id = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_id])
            
            disease_name = id_to_disease.get(predicted_id, f"Unknown_ID_{predicted_id}")
            
            # Top 3 predictions
            top3_indices = np.argsort(predictions[0])[-3:][::-1]
            top3_predictions = [{
                "disease": id_to_disease.get(int(idx), f"Unknown_{idx}"),
                "confidence": float(predictions[0][idx])
            } for idx in top3_indices]
            
            # Health status
            is_healthy = any(kw in disease_name.lower() 
                           for kw in ['healthy', 'normal', 'no_disease', 'no disease'])
            
            logger.info(f"🏥 Detection: {disease_name} ({confidence:.1%})")
            logger.info(f"   Health: {'🟢 Healthy' if is_healthy else '🔴 Disease'}")
            
            return {
                "status": "success",
                "predicted_disease": disease_name,
                "confidence": confidence,
                "is_healthy": is_healthy,
                "top3_predictions": top3_predictions,
                "model_used": self.primary_model_name,
                "total_classes": len(class_mapping)
            }
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}
    
    @property
    def is_loaded(self) -> bool:
        return self.is_available and len(self.models) > 0
    
    @property
    def available_diseases(self) -> List[str]:
        if not self.class_mappings:
            return []
        primary = self.class_mappings.get(self.primary_model_name, {})
        return sorted(primary.keys())

# ============================================================
# 🐟 SPECIES CLASSIFICATION
# ============================================================
class EnhancedFishClassifier:
    """Multi-model ensemble fish species classifier"""
    
    def __init__(self, enable_ensemble: bool = True):
        self.device = Config.DEVICE
        self.models = {}
        self.class_mappings = {}
        self.transforms = None
        self.enable_ensemble = enable_ensemble
        
        logger.info(f"🐟 Initializing Species Classification (Ensemble: {enable_ensemble})")
        self._load_all_models()
        self._setup_transforms()
    
    def _load_all_models(self):
        """Load species models"""
        available_models = []
        
        for model_name, config in Config.SPECIES_MODEL_CONFIGS.items():
            if not config['path'].exists():
                logger.info(f"  Skipping {model_name} (file not found)")
                continue
                
            try:
                checkpoint = torch.load(str(config['path']), 
                                       map_location=self.device, 
                                       weights_only=False)
                
                if isinstance(checkpoint, dict):
                    state_dict = (checkpoint.get('model_state_dict') or 
                                checkpoint.get('state_dict') or checkpoint)
                else:
                    state_dict = checkpoint
                
                class_mapping = self._load_class_mapping(checkpoint, str(config['class_mapping']))
                if not class_mapping:
                    continue
                
                model = self._create_model(len(class_mapping))
                
                try:
                    model.load_state_dict(state_dict, strict=True)
                except:
                    remapped = {f'backbone.{k}' if not k.startswith('backbone.') else k: v 
                               for k, v in state_dict.items()}
                    model.load_state_dict(remapped, strict=False)
                
                model.to(self.device).eval()
                
                self.models[model_name] = model
                self.class_mappings[model_name] = class_mapping
                available_models.append((config['priority'], model_name))
                logger.info(f"  ✓ Loaded {model_name} ({len(class_mapping)} species)")
            except Exception as e:
                logger.error(f"  ✗ Failed {model_name}: {e}")
        
        if available_models:
            available_models.sort(key=lambda x: x[0])
            self.primary_model_name = available_models[0][1]
            logger.info(f"🎯 Primary species model: {self.primary_model_name}")
            
    def _load_class_mapping(self, checkpoint: Dict, external_path: str) -> Optional[Dict]:
        """Load species mapping"""
        class_mapping = None
        
        if isinstance(checkpoint, dict):
            if 'species_mapping' in checkpoint:
                sm = checkpoint['species_mapping']
                class_mapping = sm.get('species_to_id') or sm
            elif 'class_mapping' in checkpoint:
                class_mapping = checkpoint['class_mapping']
        
        if Path(external_path).exists():
            try:
                with open(external_path, 'r', encoding='utf-8') as f:
                    external = json.load(f)
                    if 'species_to_id' in external:
                        class_mapping = external['species_to_id']
                    elif 'id_to_species' in external:
                        class_mapping = {v: int(k) for k, v in external['id_to_species'].items()}
                    else:
                        class_mapping = external
            except:
                pass
        
        return class_mapping
    
    def _create_model(self, num_classes: int) -> nn.Module:
        """Create EfficientNet model"""
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        num_features = model.classifier[1].in_features
        
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(256, num_classes)
        )
        
        class FishClassifier(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
            def forward(self, x):
                return self.backbone(x)
        
        return FishClassifier(model)
    
    def _setup_transforms(self):
        """Setup preprocessing"""
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def classify_image(self, image_path: str) -> Dict[str, Any]:
        """Classify species"""
        if not self.models:
            return {"status": "error", "message": "No models loaded"}
        
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transforms(img).unsqueeze(0).to(self.device)
            
            if not self.enable_ensemble or len(self.models) == 1:
                return self._single_model_predict(img_tensor)
            return self._ensemble_predict(img_tensor)
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
    
    def _single_model_predict(self, img_tensor: torch.Tensor) -> Dict[str, Any]:
        """Single model prediction"""
        model = self.models[self.primary_model_name]
        class_mapping = self.class_mappings[self.primary_model_name]
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        idx_to_class = {v: k for k, v in class_mapping.items()}
        species_name = idx_to_class.get(predicted.item(), f"Unknown_{predicted.item()}")
        
        top3_prob, top3_indices = torch.topk(probabilities, min(3, probabilities.size(1)))
        top3_predictions = [{
            "species": idx_to_class.get(idx.item(), f"Unknown_{idx.item()}"),
            "confidence": prob.item()
        } for prob, idx in zip(top3_prob[0], top3_indices[0])]
        
        return {
            "status": "success",
            "predicted_species": species_name,
            "confidence": confidence.item(),
            "top3_predictions": top3_predictions,
            "model_used": self.primary_model_name,
            "ensemble": False
        }
    
    def _ensemble_predict(self, img_tensor: torch.Tensor) -> Dict[str, Any]:
        """Ensemble prediction"""
        vote_counts = defaultdict(lambda: {'votes': 0, 'total_confidence': 0.0})
        
        for model_name, model in self.models.items():
            class_mapping = self.class_mappings[model_name]
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            idx_to_class = {v: k for k, v in class_mapping.items()}
            species_name = idx_to_class.get(predicted.item(), f"Unknown_{predicted.item()}")
            
            vote_counts[species_name]['votes'] += 1
            vote_counts[species_name]['total_confidence'] += confidence.item()
        
        max_votes = max(vote_counts.values(), key=lambda x: (x['votes'], x['total_confidence']))
        consensus_species = [k for k, v in vote_counts.items() if v == max_votes][0]
        
        return {
            "status": "success",
            "predicted_species": consensus_species,
            "confidence": max_votes['total_confidence'] / max_votes['votes'],
            "ensemble": True,
            "agreement": max_votes['votes'] / len(self.models)
        }
    
    @property
    def is_loaded(self) -> bool:
        return len(self.models) > 0

# ============================================================
# 📊 VISUALIZATION ENGINE
# ============================================================
class VisualizationEngine:
    """Visualization engine"""
    
    def __init__(self):
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 7)
        logger.info("🎨 Visualization Engine initialized")
    
    def create_visualization(self, plot_type: str, data: Dict, title: str, **kwargs) -> Optional[str]:
        """Create visualization"""
        try:
            if plot_type == 'bar':
                return self._create_bar_chart(data, title, 
                                              kwargs.get('xlabel', ''), 
                                              kwargs.get('ylabel', ''))
            elif plot_type == 'pie':
                return self._create_pie_chart(data, title)
            elif plot_type == 'line':
                return self._create_line_chart(data, title, 
                                               kwargs.get('xlabel', ''), 
                                               kwargs.get('ylabel', ''))
            return None
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            return None
    
    def _create_bar_chart(self, data: Dict, title: str, xlabel: str, ylabel: str) -> str:
        plt.figure(figsize=(12, 7))
        bars = plt.bar(data.keys(), data.values(), color='steelblue', alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel(xlabel or "Categories")
        plt.ylabel(ylabel or "Values")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        return self._save_plot("bar_chart")
    
    def _create_pie_chart(self, data: Dict, title: str) -> str:
        plt.figure(figsize=(10, 8))
        plt.pie(data.values(), labels=data.keys(), autopct='%1.1f%%', startangle=90)
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        return self._save_plot("pie_chart")
    
    def _create_line_chart(self, data: Dict, title: str, xlabel: str, ylabel: str) -> str:
        plt.figure(figsize=(12, 7))
        plt.plot(list(data.keys()), list(data.values()), marker='o', linewidth=2)
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel(xlabel or "X-axis")
        plt.ylabel(ylabel or "Y-axis")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return self._save_plot("line_chart")
    
    def _save_plot(self, plot_type: str) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{plot_type}_{timestamp}.png"
        filepath = Config.OUTPUTS_DIR / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        return str(filepath)

# ============================================================
# 🗄️ VECTOR DATABASE
# ============================================================
class VectorDatabaseManager:
    """Vector database operations"""
    
    def __init__(self):
        logger.info("🗄️ Initializing Vector Database...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        self.vector_db = Chroma(
            persist_directory=str(Config.VECTOR_DB_DIR),
            embedding_function=self.embeddings,
            collection_name=Config.VECTOR_DB_COLLECTION
        )
        self.document_count = self.vector_db._collection.count()
        logger.info(f"✓ Vector DB loaded: {self.document_count} documents")
        
        self.retriever = self.vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": Config.RETRIEVAL_K}
        )
    
    def search(self, query: str, k: int = None) -> List[Document]:
        try:
            if k:
                return self.vector_db.similarity_search(query, k=k)
            return self.retriever.invoke(query)
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

# ============================================================
# 💬 ADVANCED RAG CHAIN
# ============================================================
class AdvancedRAGChain:
    """RAG chain with ML integration"""
    
    def __init__(self, vector_db, fish_classifier, disease_detector, viz_engine):
        logger.info("🔗 Initializing Advanced RAG Chain...")
        
        self.vector_db = vector_db
        self.fish_classifier = fish_classifier
        self.disease_detector = disease_detector
        self.viz_engine = viz_engine
        
        self.llm = ChatGroq(
            api_key=Config.GROQ_API_KEY,
            model=Config.GROQ_MODEL,
            temperature=0.2,
            max_tokens=2048
        )
        
        self.chat_messages = []
        self._build_chain()
        logger.info("✓ RAG Chain initialized")
    
    def _build_chain(self):
        """Build RAG chain"""
        """
MeenaSetu AI - Production Chain (Part 2 - Continuation)
"""

# ... (First part with all imports and classes above) ...

# Continuation of AdvancedRAGChain class:

        system_prompt = """You are **MeenaSetu AI** 🐠 - expert in fish species identification, disease detection, and aquaculture.

**CAPABILITIES:**
- AI-powered species classification
- Disease detection and diagnosis
- Aquaculture knowledge Q&A
- Data visualization

**RULES:**
1. Use only provided context for specific facts
2. Never invent data or statistics
3. Respond naturally and conversationally
4. Mix Hindi + English where helpful
5. Use relevant emojis

**CONTEXT:**
{context}

**CONVERSATION:**
{chat_history}

**QUESTION:**
{question}

**RESPONSE:**"""
        
        prompt = ChatPromptTemplate.from_template(system_prompt)
        
        def retrieve_context(query: str) -> str:
            docs = self.vector_db.search(query)
            if not docs:
                return "No specific database information available."
            
            context_parts = []
            for doc in docs[:Config.RETRIEVAL_K]:
                source = doc.metadata.get("filename", "Unknown")
                content = doc.page_content[:600]
                context_parts.append(f"[{source}]\n{content}")
            
            return "\n\n---\n\n".join(context_parts)
        
        def format_history(messages) -> str:
            if not messages:
                return "No previous conversation."
            
            formatted = []
            for msg in messages[-8:]:
                if isinstance(msg, HumanMessage):
                    formatted.append(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    formatted.append(f"Assistant: {msg.content}")
            
            return "\n".join(formatted)
        
        self.chain = (
            {
                "context": RunnableLambda(retrieve_context),
                "chat_history": RunnableLambda(lambda x: format_history(self.chat_messages)),
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def add_to_history(self, role: str, content: str):
        if role == "user":
            self.chat_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            self.chat_messages.append(AIMessage(content=content))
    
    def invoke(self, question: str, context_override: str = None) -> str:
        try:
            if context_override:
                response = self.llm.invoke([
                    HumanMessage(content=f"Context: {context_override}\n\nQuestion: {question}")
                ])
                return response.content
            return self.chain.invoke(question)
        except Exception as e:
            logger.error(f"Chain error: {e}")
            return "I encountered an error. Please try rephrasing."
    
    def clear_history(self):
        self.chat_messages = []

# ============================================================
# 🌍 MEENASETU AI - MAIN APPLICATION
# ============================================================
class MeenasetuAI:
    """Production MeenaSetu AI with Species & Disease Detection"""
    
    def __init__(self):
        logger.info("=" * 80)
        logger.info("🐠 MEENASETU AI - PRODUCTION SYSTEM 🏥")
        logger.info("=" * 80)
        
        # Initialize components
        self.fish_classifier = EnhancedFishClassifier(enable_ensemble=True)
        self.disease_detector = FishDiseaseDetector()
        self.viz_engine = VisualizationEngine()
        self.vector_db = VectorDatabaseManager()
        self.rag_chain = AdvancedRAGChain(
            self.vector_db,
            self.fish_classifier,
            self.disease_detector,
            self.viz_engine
        )
        
        # Document processor
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        # Session statistics
        self.session_stats = {
            "start_time": datetime.now().isoformat(),
            "queries_processed": 0,
            "images_classified": 0,
            "diseases_detected": 0,
            "visualizations_created": 0,
            "documents_uploaded": 0
        }
        
        logger.info("✅ MeenaSetu AI fully initialized!")
        logger.info(f"🐟 Species Models: {len(self.fish_classifier.models)}")
        logger.info(f"🏥 Disease Models: {len(self.disease_detector.models) if self.disease_detector.is_loaded else 0}")
        logger.info(f"📚 Vector DB: {self.vector_db.document_count} documents")
        
        if not self.fish_classifier.is_loaded:
            logger.warning("⚠️ Species models not loaded")
        
        if not self.disease_detector.is_loaded:
            logger.warning("⚠️ Disease models not loaded")
    
    def process_query(self, query: str, image_path: str = None) -> Dict[str, Any]:
        """Process user query with optional image"""
        logger.info(f"❓ Processing: {query[:80]}...")
        
        result = {
            "query": query,
            "answer": "",
            "image_classification": None,
            "disease_detection": None,
            "visualization": None,
            "sources": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Handle image if provided
        if image_path and Path(image_path).exists():
            image_result = self._process_image(image_path, query)
            result.update(image_result)
        
        # Generate answer with context
        context_addition = ""
        
        if result.get('image_classification'):
            ic = result['image_classification']
            context_addition += f"\n\n**Species:** {ic['predicted_species']} ({ic['confidence']:.1%})"
        
        if result.get('disease_detection'):
            dd = result['disease_detection']
            if dd['status'] == 'success':
                health = "🟢 Healthy" if dd.get('is_healthy') else "🔴 Disease"
                context_addition += f"\n\n**Health:** {health}\n{dd['predicted_disease']} ({dd['confidence']:.1%})"
        
        # Add to history and get answer
        self.rag_chain.add_to_history("user", query)
        
        if context_addition:
            answer = self.rag_chain.invoke(query, context_override=context_addition)
        else:
            answer = self.rag_chain.invoke(query)
        
        result['answer'] = answer
        self.rag_chain.add_to_history("assistant", answer)
        
        # Update stats
        self.session_stats['queries_processed'] += 1
        if result.get('image_classification'):
            self.session_stats['images_classified'] += 1
        if result.get('disease_detection') and result['disease_detection']['status'] == 'success':
            self.session_stats['diseases_detected'] += 1
        
        logger.info("✅ Query processed successfully")
        return result
    
    def _process_image(self, image_path: str, query: str) -> Dict:
        """Process image for species classification AND disease detection"""
        result = {}
        
        # Species classification
        if self.fish_classifier.is_loaded:
            classification = self.fish_classifier.classify_image(image_path)
            if classification['status'] == 'success':
                result['image_classification'] = classification
                logger.info(f"🐟 Species: {classification['predicted_species']} ({classification['confidence']:.1%})")
        
        # Disease detection
        disease_keywords = ['disease', 'sick', 'unhealthy', 'symptoms', 'treatment', 'health']
        should_check_disease = any(kw in query.lower() for kw in disease_keywords)
        
        if self.disease_detector.is_loaded and (should_check_disease or result.get('image_classification')):
            disease_result = self.disease_detector.detect_disease(image_path)
            if disease_result['status'] == 'success':
                result['disease_detection'] = disease_result
                health = "Healthy" if disease_result.get('is_healthy') else disease_result['predicted_disease']
                logger.info(f"🏥 Health: {health} ({disease_result['confidence']:.1%})")
        
        return result
    
    def upload_document(self, file_path: str) -> Dict:
        """Upload and process document"""
        try:
            ext = Path(file_path).suffix.lower()
            
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            elif ext == '.csv':
                df = pd.read_csv(file_path)
                content = f"CSV Data: {Path(file_path).name}\n\n{df.to_string(index=False)}"
                docs = [Document(page_content=content, metadata={
                    "source": file_path,
                    "type": "csv",
                    "filename": Path(file_path).name
                })]
            elif ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                docs = [Document(page_content=content, metadata={
                    "source": file_path,
                    "type": "txt",
                    "filename": Path(file_path).name
                })]
            elif ext in ['.jpg', '.jpeg', '.png']:
                img_result = self._process_image(file_path, "")
                
                content = f"Image: {Path(file_path).name}\n"
                if img_result.get('image_classification'):
                    ic = img_result['image_classification']
                    content += f"Species: {ic['predicted_species']} ({ic['confidence']:.1%})\n"
                
                if img_result.get('disease_detection'):
                    dd = img_result['disease_detection']
                    content += f"Health: {dd['predicted_disease']} ({dd['confidence']:.1%})\n"
                
                docs = [Document(page_content=content, metadata={
                    "source": file_path,
                    "type": "image",
                    "filename": Path(file_path).name,
                    **img_result
                })]
            else:
                return {"status": "error", "message": f"Unsupported file type: {ext}"}
            
            # Split and add to vector DB
            chunks = self.text_splitter.split_documents(docs)
            self.vector_db.vector_db.add_documents(chunks)
            
            self.session_stats['documents_uploaded'] += 1
            
            return {
                "status": "success",
                "filename": Path(file_path).name,
                "chunks_created": len(chunks),
                "message": f"Successfully uploaded {Path(file_path).name}"
            }
        except Exception as e:
            logger.error(f"Document upload error: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        return {
            "session": self.session_stats,
            "database": {
                "total_documents": self.vector_db.document_count,
                "collection": Config.VECTOR_DB_COLLECTION
            },
            "ml_models": {
                "species_models": len(self.fish_classifier.models),
                "disease_models": len(self.disease_detector.models) if self.disease_detector.is_loaded else 0,
                "species_loaded": self.fish_classifier.is_loaded,
                "disease_loaded": self.disease_detector.is_loaded,
                "ensemble_enabled": self.fish_classifier.enable_ensemble,
                "tensorflow_available": TENSORFLOW_AVAILABLE,
                "device": str(Config.DEVICE)
            },
            "conversation": {
                "messages": len(self.rag_chain.chat_messages)
            }
        }
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.rag_chain.clear_history()
        logger.info("🧹 Conversation history cleared")

# ============================================================
# 🧪 TESTING & DEMONSTRATION
# ============================================================
def main():
    """Production demonstration"""
    print("\n" + "=" * 80)
    print("🐠 MEENASETU AI - PRODUCTION SYSTEM")
    print("=" * 80 + "\n")
    
    try:
        # Initialize
        ai = MeenasetuAI()
        
        # Test 1: General query
        print("\n📝 TEST 1: General Aquatic Query")
        print("-" * 80)
        result = ai.process_query("What are the main fish species in West Bengal?")
        print(f"Answer: {result['answer'][:200]}...")
        
        # Test 2: Follow-up
        print("\n\n🔄 TEST 2: Follow-up Question")
        print("-" * 80)
        result = ai.process_query("Which one is most commercially important?")
        print(f"Answer: {result['answer'][:200]}...")
        
        # Test 3: Statistics
        print("\n\n📊 TEST 3: System Statistics")
        print("-" * 80)
        stats = ai.get_statistics()
        print(f"✓ Queries Processed: {stats['session']['queries_processed']}")
        print(f"✓ Database Documents: {stats['database']['total_documents']}")
        print(f"✓ Species Models: {stats['ml_models']['species_models']}")
        print(f"✓ Disease Models: {stats['ml_models']['disease_models']}")
        print(f"✓ Species Loaded: {stats['ml_models']['species_loaded']}")
        print(f"✓ Disease Loaded: {stats['ml_models']['disease_loaded']}")
        print(f"✓ TensorFlow: {stats['ml_models']['tensorflow_available']}")
        print(f"✓ Device: {stats['ml_models']['device']}")
        
        # Test 4: Image capabilities
        print("\n\n🖼️ TEST 4: Image Processing Capabilities")
        print("-" * 80)
        print(f"Species Classification: {'✅ Ready' if ai.fish_classifier.is_loaded else '❌ Not loaded'}")
        print(f"Disease Detection: {'✅ Ready' if ai.disease_detector.is_loaded else '❌ Not loaded'}")
        
        if ai.disease_detector.is_loaded:
            diseases = ai.disease_detector.available_diseases
            print(f"Available Diseases: {len(diseases)}")
            if diseases:
                print(f"  Sample: {', '.join(diseases[:5])}")
        
        print("\n" + "=" * 80)
        print("✨ PRODUCTION DEMO COMPLETE ✨")
        print("=" * 80 + "\n")
        
        print("\n📋 FEATURES AVAILABLE:")
        features = [
            ("🐟 Species Classification", f"Ensemble of {len(ai.fish_classifier.models)} models"),
            ("🏥 Disease Detection", f"{len(ai.disease_detector.models) if ai.disease_detector.is_loaded else 0} Keras models"),
            ("💬 RAG-based Q&A", "Conversational memory"),
            ("📊 Data Visualization", "Plotly, Matplotlib"),
            ("📁 Document Processing", "PDF, CSV, Images"),
            ("🔍 Semantic Search", f"{ai.vector_db.document_count} documents")
        ]
        
        for feature, detail in features:
            print(f"  {feature}: {detail}")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()