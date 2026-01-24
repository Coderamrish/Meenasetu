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
# Keras 3 for Disease Detection (NOT tf_keras)
try:
    import keras  # <-- FIXED: Use keras instead of tensorflow.keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("⚠️ Keras not available - disease detection disabled")
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

# LOGGING & ENVIRONMENT
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

#  CONFIGURATION
class Config:
    SCRIPT_DIR = Path(__file__).resolve().parent
    BASE_DIR = SCRIPT_DIR.parent.parent.parent
    # Verify BASE_DIR
    if not (BASE_DIR / "training").exists():
        BASE_DIR = Path.cwd()
        if not (BASE_DIR / "training").exists():
            current = SCRIPT_DIR
            for _ in range(5):
                if (current / "training").exists():
                    BASE_DIR = current
                    break
                current = current.parent
    
    # Directories
    VECTOR_DB_DIR = BASE_DIR / "models" / "vector_db"
    UPLOADS_DIR = BASE_DIR / "backend" / "uploads"
    OUTPUTS_DIR = BASE_DIR / "backend" / "outputs"
    CACHE_DIR = BASE_DIR / "backend" / "cache"
    TRAINING_DIR = BASE_DIR / "training"
    
    # Species Classification Models
    SPECIES_MODEL_CONFIGS = {
        'fish_species_model': {
            'path': TRAINING_DIR / "checkpoints" / "fish_model.pth",
            'class_mapping': TRAINING_DIR / "checkpoints" / "class_mapping1.json",
            'priority': 1
        },
        'best_model': {
            'path': TRAINING_DIR / "checkpoints" / "best_model.pth",
            'class_mapping': TRAINING_DIR / "checkpoints" / "class_mapping.json",
            'priority': 2
        },
        'final_model': {
            'path': TRAINING_DIR / "checkpoints" / "final_model.pth",
            'class_mapping': TRAINING_DIR / "checkpoints" / "class_mapping.json",
            'priority': 3
        }
    }
    
    # Disease Detection Models (Keras 3)
    DISEASE_MODEL_CONFIGS = {
        'disease_model_final': {
            'path': TRAINING_DIR / "checkpoints" / "best_efficientnet_freshwater.keras",
            'class_mapping': TRAINING_DIR / "checkpoints" / "disease_class_mapping.json",
            'priority': 1
        },
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
        for directory in [Config.UPLOADS_DIR, Config.OUTPUTS_DIR, Config.CACHE_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

Config.setup_directories()

# FIXED DISEASE DETECTION SYSTEM
class FishDiseaseDetector:
    def __init__(self):
        self.models = {}
        self.class_mappings = {}
        self.id_to_disease_maps = {}
        self.primary_model_name = None
        self.is_available = KERAS_AVAILABLE
        
        if not self.is_available:
            logger.warning(" Disease Detection DISABLED (Keras not available)")
            return
        
        logger.info(" Initializing Fish Disease Detection System")
        self._load_all_models()
    
    def _load_class_mapping(self, mapping_path: Path) -> Optional[Dict[str, int]]:
        try:
            if not mapping_path.exists():
                logger.error(f"   ✗ Mapping file not found: {mapping_path}")
                return None
            
            with open(str(mapping_path), 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            logger.info(f"   Loaded JSON from {mapping_path.name}")
            logger.info(f"   Raw data type: {type(raw_data)}")
            
            # STRATEGY 1: Direct dict format {"disease_name": id}
            if isinstance(raw_data, dict):
                # Check if values are integers (disease -> id format)
                if all(isinstance(v, int) for v in raw_data.values()):
                    logger.info("    Format: disease_name -> id")
                    return {str(k): int(v) for k, v in raw_data.items()}
                
                # Check if keys are numeric strings (id -> disease format)
                if all(str(k).isdigit() for k in raw_data.keys()):
                    logger.info("    Format: id -> disease_name (converting)")
                    # Convert {"0": "disease", "1": "disease"} to {"disease": 0}
                    result = {}
                    for id_str, disease_name in raw_data.items():
                        result[str(disease_name)] = int(id_str)
                    logger.info(f"    Converted {len(result)} mappings")
                    return result
                
                # Check for nested structures
                if 'disease_to_id' in raw_data:
                    logger.info("    Format: nested disease_to_id")
                    return {str(k): int(v) for k, v in raw_data['disease_to_id'].items()}
                
                if 'id_to_disease' in raw_data:
                    logger.info("    Format: nested id_to_disease (converting)")
                    result = {}
                    for id_str, disease_name in raw_data['id_to_disease'].items():
                        result[str(disease_name)] = int(id_str)
                    return result
            
            # STRATEGY 2: List format [diseases...]
            if isinstance(raw_data, list):
                logger.info("    Format: list of diseases")
                return {str(disease): idx for idx, disease in enumerate(raw_data)}
            
            logger.error("    Unknown JSON format")
            logger.error(f"   Sample data: {str(raw_data)[:200]}")
            return None
                
        except json.JSONDecodeError as e:
            logger.error(f"    JSON parse error: {e}")
            return None
        except Exception as e:
            logger.error(f"    Error loading mapping: {e}")
            import traceback
            logger.error(f"   {traceback.format_exc()}")
            return None
    
    def _load_all_models(self):
        available_models = []
        
        for model_name, config in Config.DISEASE_MODEL_CONFIGS.items():
            model_path = config['path']
            mapping_path = config['class_mapping']
            
            logger.info(f"\n{'='*60}")
            logger.info(f" Loading: {model_name}")
            logger.info(f"{'='*60}")
            logger.info(f"   Model: {model_path.name}")
            logger.info(f"   Path: {model_path}")
            logger.info(f"   Exists: {model_path.exists()}")
            logger.info(f"   Mapping: {mapping_path.name}")
            logger.info(f"   Exists: {mapping_path.exists()}")
            
            if not model_path.exists():
                logger.warning(f"   Model file missing: {model_path}")
                continue
            
            if not mapping_path.exists():
                logger.warning(f"   Mapping file missing: {mapping_path}")
                continue
            
            try:
                # STEP 1: Load class mapping
                logger.info(f"   Step 1: Loading class mapping...")
                with open(str(mapping_path), 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                # Convert to disease->id format
                class_mapping = {}
                if isinstance(raw_data, dict):
                    if all(str(k).isdigit() for k in raw_data.keys()):
                        # Format: {"0": "disease1", "1": "disease2"}
                        for id_str, disease_name in raw_data.items():
                            class_mapping[str(disease_name)] = int(id_str)
                    else:
                        # Format: {"disease1": 0, "disease2": 1}
                        class_mapping = {str(k): int(v) for k, v in raw_data.items()}
                
                logger.info(f"   Loaded {len(class_mapping)} disease classes")
                
                # STEP 2: Load Keras 3 model
                logger.info(f"   Step 2: Loading Keras model...")
                model = keras.models.load_model(str(model_path), compile=False)
                logger.info(f"  Model loaded")
                logger.info(f"   Input shape: {model.input_shape}")
                logger.info(f"   Output shape: {model.output_shape}")
                
                # STEP 3: Verify dimensions
                model_classes = model.output_shape[-1]
                mapping_classes = len(class_mapping)
                
                if model_classes != mapping_classes:
                    logger.error(f"   DIMENSION MISMATCH!")
                    logger.error(f"    Model outputs: {model_classes} classes")
                    logger.error(f"    Mapping has: {mapping_classes} classes")
                    logger.error(f"   Skipping this model")
                    continue
                
                # STEP 4: Create reverse mapping
                id_to_disease = {v: k for k, v in class_mapping.items()}
                
                # STEP 5: Store everything
                self.models[model_name] = model
                self.class_mappings[model_name] = class_mapping
                self.id_to_disease_maps[model_name] = id_to_disease
                available_models.append((config['priority'], model_name))
                
                logger.info(f"   FULLY LOADED: {model_name}")
                logger.info(f"  {'='*60}\n")
                
            except Exception as e:
                logger.error(f"   EXCEPTION while loading {model_name}")
                logger.error(f"  Error: {str(e)}")
                import traceback
                logger.error(f"  Traceback:\n{traceback.format_exc()}")
                logger.info(f"  {'='*60}\n")
        
        # Set primary model
        if available_models:
            available_models.sort(key=lambda x: x[0])
            self.primary_model_name = available_models[0][1]
            
            logger.info(f"\n{'='*70}")
            logger.info(f" DISEASE DETECTION SYSTEM READY")
            logger.info(f"{'='*70}")
            logger.info(f" Primary model: {self.primary_model_name}")
            logger.info(f" Total models loaded: {len(self.models)}")
            logger.info(f" Disease classes: {len(self.class_mappings[self.primary_model_name])}")
            logger.info(f"{'='*70}\n")
        else:
            logger.error(f"\n{'='*70}")
            logger.error(" NO DISEASE MODELS LOADED")
            logger.error(f"{'='*70}\n")
    
    def detect_disease(self, image_path: str) -> Dict[str, Any]:
        if not self.is_available or not self.models:
            return {
                "status": "unavailable", 
                "message": "Disease detection not available"
            }
        
        try:
            img = Image.open(image_path).convert('RGB').resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            model = self.models[self.primary_model_name]
            id_to_disease = self.id_to_disease_maps[self.primary_model_name]
            
            predictions = model.predict(img_array, verbose=0)
            predicted_id = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_id])
            
            # Get disease name
            disease_name = id_to_disease.get(predicted_id, f"Unknown_ID_{predicted_id}")
            
            # Get top 3 predictions
            top3_indices = np.argsort(predictions[0])[-3:][::-1]
            top3_predictions = [
                {
                    "disease": id_to_disease.get(int(idx), f"Unknown_{idx}"),
                    "confidence": float(predictions[0][idx])
                }
                for idx in top3_indices
            ]
            
            healthy_keywords = ['healthy', 'normal', 'no disease', 'no_disease', 'good', 'fine']
            
            # Separate disease vs healthy predictions
            disease_predictions = [
                p for p in top3_predictions 
                if not any(kw in p['disease'].lower() for kw in healthy_keywords)
            ]
            
            healthy_predictions = [
                p for p in top3_predictions
                if any(kw in p['disease'].lower() for kw in healthy_keywords)
            ]
            
            # Calculate total confidence for each category
            total_disease_conf = sum(p['confidence'] for p in disease_predictions)
            total_healthy_conf = sum(p['confidence'] for p in healthy_predictions)
            
            logger.info(f" Voting Analysis:")
            logger.info(f"   Disease votes: {len(disease_predictions)} predictions (combined: {total_disease_conf:.1%})")
            logger.info(f"   Healthy votes: {len(healthy_predictions)} predictions (combined: {total_healthy_conf:.1%})")
            logger.info(f"   Top prediction: {disease_name} ({confidence:.1%})")
            
            # THRESHOLDS
            HIGH_CONFIDENCE = 0.60              
            MEDIUM_CONFIDENCE = 0.45            
            COMBINED_DISEASE_THRESHOLD = 0.50   
            STRONG_DISEASE_SIGNAL = 0.40        
            
            # CASE 1: Strong combined disease signal
            if total_disease_conf >= COMBINED_DISEASE_THRESHOLD:
            
                top_disease = max(disease_predictions, key=lambda x: x['confidence'])
                
                is_actually_healthy = False
                final_diagnosis = top_disease['disease']
                final_confidence = top_disease['confidence']
                
                logger.warning(
                    f" DISEASE DETECTED (voting): Multiple disease signals "
                    f"(combined: {total_disease_conf:.1%}) - "
                    f"Top disease: {final_diagnosis} ({final_confidence:.1%})"
                )
            
            # CASE 2: No healthy predictions AND reasonable disease signal
            elif len(healthy_predictions) == 0 and total_disease_conf >= STRONG_DISEASE_SIGNAL:
                # All top-3 are diseases AND they sum to >40%
                top_disease = disease_predictions[0]  # Highest disease
                
                is_actually_healthy = False
                final_diagnosis = top_disease['disease']
                final_confidence = top_disease['confidence']
                
                logger.warning(
                    f" DISEASE DETECTED (unanimous): No healthy predictions, "
                    f"diseases at {total_disease_conf:.1%} - "
                    f"Reporting: {final_diagnosis} ({final_confidence:.1%})"
                )
            
            # CASE 3: High confidence single prediction
            elif confidence >= HIGH_CONFIDENCE:
                is_predicted_healthy = any(kw in disease_name.lower() for kw in healthy_keywords)
                is_actually_healthy = is_predicted_healthy
                final_diagnosis = disease_name
                final_confidence = confidence
                
                logger.info(f" High confidence single prediction ({confidence:.1%}): {final_diagnosis}")
            
            # CASE 4: Medium confidence single prediction
            elif confidence >= MEDIUM_CONFIDENCE:
                is_predicted_healthy = any(kw in disease_name.lower() for kw in healthy_keywords)
                
                if is_predicted_healthy:
                    # Predicted healthy - check if diseases are competing
                    if disease_predictions and disease_predictions[0]['confidence'] > 0.35:
                        gap = confidence - disease_predictions[0]['confidence']
                        
                        if gap < 0.10:  # Less than 10% gap
                            # Too close - report disease to be safe
                            is_actually_healthy = False
                            final_diagnosis = disease_predictions[0]['disease']
                            final_confidence = disease_predictions[0]['confidence']
                            logger.warning(
                                f" Close call: Healthy {confidence:.1%} vs "
                                f"{final_diagnosis} {final_confidence:.1%} - "
                                f"Reporting disease to be safe"
                            )
                        else:
                            # Healthy clearly stronger
                            is_actually_healthy = True
                            final_diagnosis = "healthy"
                            final_confidence = confidence
                            logger.info(f" Likely healthy ({confidence:.1%})")
                    else:
                        # No strong disease signal
                        is_actually_healthy = True
                        final_diagnosis = "healthy"
                        final_confidence = confidence
                        logger.info(f" Healthy ({confidence:.1%})")
                else:
                    # Predicted disease with medium confidence
                    is_actually_healthy = False
                    final_diagnosis = disease_name
                    final_confidence = confidence
                    logger.info(f" Disease detected: {final_diagnosis} ({confidence:.1%})")
            
            # CASE 5: Low confidence - use voting
            else:
                if total_disease_conf > total_healthy_conf and total_disease_conf > 0.38:
                    # Diseases collectively stronger AND above minimum threshold
                    top_disease = max(disease_predictions, key=lambda x: x['confidence'])
                    
                    is_actually_healthy = False
                    final_diagnosis = top_disease['disease']
                    final_confidence = top_disease['confidence']
                    
                    logger.warning(
                        f" Low individual confidence but diseases collectively strong "
                        f"(combined: {total_disease_conf:.1%}) - "
                        f"Reporting: {final_diagnosis} ({final_confidence:.1%})"
                    )
                else:
                    # Default to healthy
                    is_actually_healthy = True
                    final_diagnosis = "healthy"
                    final_confidence = confidence
                    
                    logger.info(
                        f" Defaulting to healthy - weak signals "
                        f"(disease: {total_disease_conf:.1%}, healthy: {total_healthy_conf:.1%})"
                    )
            
            result = {
                "status": "success",
                "predicted_disease": final_diagnosis,
                "confidence": final_confidence,
                "is_healthy": is_actually_healthy,
                "top3_predictions": top3_predictions,
                "model_used": self.primary_model_name,
                "total_classes": len(id_to_disease),
                "voting_stats": {
                    "total_disease_confidence": total_disease_conf,
                    "total_healthy_confidence": total_healthy_conf,
                    "disease_votes": len(disease_predictions),
                    "healthy_votes": len(healthy_predictions)
                }
            }
            
            # Add confidence notes
            if final_confidence >= HIGH_CONFIDENCE:
                result['confidence_note'] = "HIGH - Reliable prediction"
            elif final_confidence >= MEDIUM_CONFIDENCE:
                result['confidence_note'] = "MEDIUM - Good prediction"
            elif total_disease_conf >= COMBINED_DISEASE_THRESHOLD:
                result['confidence_note'] = f"VOTING-BASED - Multiple disease signals ({total_disease_conf:.0%})"
            else:
                result['confidence_note'] = "LOW - Verify recommended"
            
            # Add warnings for edge cases
            if total_disease_conf >= COMBINED_DISEASE_THRESHOLD and final_confidence < MEDIUM_CONFIDENCE:
                result['warning'] = (
                    f"Disease detected based on voting (combined confidence: {total_disease_conf:.1%}). "
                    f"Individual prediction confidence is {final_confidence:.1%}. "
                    f"Manual verification recommended."
                )
            elif is_actually_healthy and total_disease_conf > 0.40:
                result['warning'] = (
                    f"Fish appears healthy, but disease signals detected at {total_disease_conf:.1%}. "
                    f"Monitor closely for next 24-48 hours."
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "status": "error", 
                "message": str(e)
            }
    
    @property
    def is_loaded(self) -> bool:
        return self.is_available and len(self.models) > 0
    
    @property
    def available_diseases(self) -> List[str]:
        if not self.class_mappings:
            return []
        primary = self.class_mappings.get(self.primary_model_name, {})
        return sorted(primary.keys())
#  SPECIES CLASSIFICATION
class EnhancedFishClassifier:
    
    def __init__(self, enable_ensemble: bool = True):
        self.device = Config.DEVICE
        self.models = {}
        self.class_mappings = {}
        self.transforms = None
        self.enable_ensemble = enable_ensemble
        
        logger.info(f" Initializing Species Classification (Ensemble: {enable_ensemble})")
        self._load_all_models()
        self._setup_transforms()
    
    def _load_all_models(self):
        """Load species models"""
        available_models = []
        
        for model_name, config in Config.SPECIES_MODEL_CONFIGS.items():
            if not config['path'].exists():
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
                logger.info(f" Loaded {model_name} ({len(class_mapping)} species)")
            except Exception as e:
                logger.error(f" Failed {model_name}: {e}")
        
        if available_models:
            available_models.sort(key=lambda x: x[0])
            self.primary_model_name = available_models[0][1]
            logger.info(f" Primary species model: {self.primary_model_name}")
    
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
#  VISUALIZATION ENGINE

class VisualizationEngine:
    """Visualization engine"""
    
    def __init__(self):
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 7)
        logger.info(" Visualization Engine initialized")
    
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


# VECTOR DATABASE
class VectorDatabaseManager:
    """Vector database operations"""
    
    def __init__(self):
        logger.info(" Initializing Vector Database...")
        
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
        logger.info(f" Vector DB loaded: {self.document_count} documents")
        
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
# ADVANCED RAG CHAIN

class AdvancedRAGChain:
    """RAG chain with ML integration"""
    
    def __init__(self, vector_db, fish_classifier, disease_detector, viz_engine):
        logger.info(" Initializing Advanced RAG Chain...")
        
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
        system_prompt = """You are **MeenaSetu AI** 🐠 - India's intelligent aquatic assistant for farmers, students, and researchers.

## 🎯 YOUR EXPERTISE
- **Species Identification:** Recognize fish from descriptions/images
- **Disease Detection:** Identify common fish diseases and symptoms  
- **Aquaculture Guidance:** Practical farming advice for Indian conditions
- **Health Monitoring:** Preventive care and early warning signs
- **Regional Knowledge:** South Asian freshwater ecosystems

## 📊 KNOWLEDGE SOURCES
You have access to:
1. **Provided Context:** {context} (ALWAYS check this first)
2. **General Aquatic Knowledge:** Common fish species, widespread diseases
3. **Scientific Principles:** Basic aquaculture, fish biology, water chemistry

## 🚨 CRITICAL RULES
1. **CONTEXT IS KING:** For specific facts, names, statistics → MUST use {context}
2. **GENERAL KNOWLEDGE OK:** For common species/diseases not in context, use your knowledge but say: "Generally, [common knowledge]"
3. **NO INVENTION:** Never create fake species, diseases, or data
4. **UNCERTAINTY IS OK:** If unsure → "Confirmed information not available"
5. **SAFETY FIRST:**  medication dosages only after doctor consult → Only preventive measures

## 🗣️ COMMUNICATION STYLE
- **Language:** Always use English as language
- **Tone:** Friendly, practical, non-alarming
- **Clarity:** Explain simply, avoid jargon
- **Emojis:** Use 🐟🌊💊📊🔬 where helpful
- **Farmer-Friendly:** "Aap yeh try kar sakte hain" not "Implement this protocol"

## 🏥 DISEASE GUIDELINES
- If disease in {context} → Use that detailed information
- If common disease (ich, fin rot, dropsy) not in {context} → Use general knowledge
- **Always add:** "For serious cases, local fisheries officer se consult karein"

## 🐟 SPECIES GUIDELINES  
- If species in {context} → Use local names & details from context
- If common Indian species (rohu, katla, magur) not in {context} → Use general knowledge
- **Always mention:** "Pehchanne ke liye [key feature] dekhein"

## 📦 PROVIDED INFORMATION
{context}

## 💬 CONVERSATION HISTORY
{chat_history}

## ❓ USER'S QUESTION
{question}

## ✍️ YOUR RESPONSE
[Start directly. Be helpful. Mix languages naturally. End with practical next step.]"""
        
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
# MEENASETU AI - MAIN APPLICATION
class MeenasetuAI:
    """Production MeenaSetu AI with Species & Disease Detection"""
    
    def __init__(self):
        logger.info("=" * 80)
        logger.info(" MEENASETU AI - PRODUCTION SYSTEM 🏥")
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
        
        logger.info("=" * 80)
        logger.info(" MEENASETU AI INITIALIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f" Species Models: {len(self.fish_classifier.models)}")
        logger.info(f" Disease Models: {len(self.disease_detector.models) if self.disease_detector.is_loaded else 0}")
        logger.info(f" Vector DB: {self.vector_db.document_count} documents")
        logger.info(f" Device: {Config.DEVICE}")
        logger.info("=" * 80)
        
        if not self.fish_classifier.is_loaded:
            logger.warning(" Species classification unavailable")
        
        if not self.disease_detector.is_loaded:
            logger.warning(" Disease detection unavailable")
    
    def process_query(self, query: str, image_path: str = None) -> Dict[str, Any]:
        """Process user query with optional image"""
        logger.info(f" Processing query: {query[:80]}...")
        
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
            logger.info(f" Processing image: {Path(image_path).name}")
            image_result = self._process_image(image_path, query)
            result.update(image_result)
        
        # Generate answer with context
        context_addition = ""
        
        if result.get('image_classification'):
            ic = result['image_classification']
            context_addition += f"\n\n**Species Classification:**\n"
            context_addition += f"- Predicted: {ic['predicted_species']}\n"
            context_addition += f"- Confidence: {ic['confidence']:.1%}\n"
        
        if result.get('disease_detection'):
            dd = result['disease_detection']
            if dd['status'] == 'success':
                health_status = " Healthy" if dd.get('is_healthy') else " Disease Detected"
                context_addition += f"\n**Disease Detection:**\n"
                context_addition += f"- Status: {health_status}\n"
                context_addition += f"- Diagnosis: {dd['predicted_disease']}\n"
                context_addition += f"- Confidence: {dd['confidence']:.1%}\n"
        
        # Add to history and get answer
        self.rag_chain.add_to_history("user", query)
        
        if context_addition:
            full_context = f"{context_addition}\n\nUser Question: {query}"
            answer = self.rag_chain.invoke(query, context_override=full_context)
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
        
        logger.info(" Query processed successfully")
        return result
    
    def _process_image(self, image_path: str, query: str) -> Dict:
        """Process image for species classification AND disease detection"""
        result = {}
        
        # Species classification
        if self.fish_classifier.is_loaded:
            logger.info("   Running species classification...")
            classification = self.fish_classifier.classify_image(image_path)
            if classification['status'] == 'success':
                result['image_classification'] = classification
                logger.info(f"   Species: {classification['predicted_species']} "
                          f"({classification['confidence']:.1%})")
        
        # Disease detection
        disease_keywords = ['disease', 'sick', 'unhealthy', 'symptoms', 'treatment', 
                          'health', 'diagnosis', 'infected', 'infection']
        should_check_disease = any(kw in query.lower() for kw in disease_keywords)
        
        # Always check disease if we have a species classification OR if query mentions health
        if self.disease_detector.is_loaded and (should_check_disease or result.get('image_classification')):
            logger.info("   Running disease detection...")
            disease_result = self.disease_detector.detect_disease(image_path)
            if disease_result['status'] == 'success':
                result['disease_detection'] = disease_result
                health = ("Healthy" if disease_result.get('is_healthy') 
                         else disease_result['predicted_disease'])
                logger.info(f"   Health: {health} ({disease_result['confidence']:.1%})")
        
        return result
    
    def upload_document(self, file_path: str) -> Dict:
        """Upload and process document"""
        try:
            ext = Path(file_path).suffix.lower()
            logger.info(f" Uploading document: {Path(file_path).name} ({ext})")
            
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
            
            logger.info(f"✓ Uploaded {Path(file_path).name} ({len(chunks)} chunks)")
            
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
                "keras_available": KERAS_AVAILABLE,
                "device": str(Config.DEVICE),
                "available_diseases": (self.disease_detector.available_diseases 
                                     if self.disease_detector.is_loaded else [])
            },
            "conversation": {
                "messages": len(self.rag_chain.chat_messages)
            }
        }
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history as list of dicts"""
        history = []
        for msg in self.rag_chain.chat_messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        return history
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.rag_chain.clear_history()
        logger.info(" Conversation history cleared")
#  TESTING & DEMONSTRATION
def main():
    """Production demonstration"""
    print("\n" + "=" * 80)
    print(" MEENASETU AI - FIXED PRODUCTION SYSTEM TEST")
    print("=" * 80 + "\n")
    
    try:
        # Initialize
        ai = MeenasetuAI()
        
        # Test 1: General query
        print("\n" + "=" * 80)
        print(" TEST 1: General Aquatic Query")
        print("=" * 80)
        result = ai.process_query("What are the main fish species in West Bengal?")
        print(f"\n Answer:\n{result['answer'][:300]}...\n")
        
        # Test 2: Follow-up
        print("\n" + "=" * 80)
        print(" TEST 2: Follow-up Question")
        print("=" * 80)
        result = ai.process_query("Which one is most commercially important?")
        print(f"\n Answer:\n{result['answer'][:300]}...\n")
        
        # Test 3: System Statistics
        print("\n" + "=" * 80)
        print(" TEST 3: System Statistics")
        print("=" * 80)
        stats = ai.get_statistics()
        
        print(f"\n Session Stats:")
        print(f"  • Queries Processed: {stats['session']['queries_processed']}")
        print(f"  • Images Classified: {stats['session']['images_classified']}")
        print(f"  • Diseases Detected: {stats['session']['diseases_detected']}")
        
        print(f"\n Database:")
        print(f"  • Total Documents: {stats['database']['total_documents']}")
        print(f"  • Collection: {stats['database']['collection']}")
        
        print(f"\n ML Models:")
        print(f"  • Species Models: {stats['ml_models']['species_models']}")
        print(f"  • Disease Models: {stats['ml_models']['disease_models']}")
        print(f"  • Species Loaded: {'' if stats['ml_models']['species_loaded'] else '❌'}")
        print(f"  • Disease Loaded: {'' if stats['ml_models']['disease_loaded'] else '❌'}")
        print(f"  • Keras: {'' if stats['ml_models']['keras_available'] else '❌'}")
        print(f"  • Device: {stats['ml_models']['device']}")
        
        if stats['ml_models']['available_diseases']:
            print(f"\n Available Diseases ({len(stats['ml_models']['available_diseases'])}):")
            for i, disease in enumerate(stats['ml_models']['available_diseases'][:10], 1):
                print(f"  {i}. {disease}")
            if len(stats['ml_models']['available_diseases']) > 10:
                print(f"  ... and {len(stats['ml_models']['available_diseases']) - 10} more")
        
        # Test 4: Feature Summary
        print("\n" + "=" * 80)
        print(" FEATURE SUMMARY")
        print("=" * 80)
        
        features = [
            (" Species Classification", 
             f"{' ' if ai.fish_classifier.is_loaded else ' '}"
             f"Ensemble of {len(ai.fish_classifier.models)} models"),
            
            (" Disease Detection", 
             f"{' ' if ai.disease_detector.is_loaded else ' '}"
             f"{len(ai.disease_detector.models) if ai.disease_detector.is_loaded else 0} Keras models"),
            
            (" RAG-based Q&A", 
             " Conversational memory with context"),
            
            (" Data Visualization", 
             " Plotly, Matplotlib charts"),
            
            (" Document Processing", 
             " PDF, CSV, TXT, Images"),
            
            (" Semantic Search", 
             f" {ai.vector_db.document_count} documents indexed")
        ]
        
        print()
        for feature, status in features:
            print(f"  {feature:.<40} {status}")
        
        print("\n" + "=" * 80)
        print(" PRODUCTION DEMO COMPLETE")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()