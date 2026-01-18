import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import base64
import hashlib
from collections import defaultdict
import pickle

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from tqdm import tqdm

# Import TensorFlow for Keras models
import tensorflow as tf
from tensorflow import keras

# ENHANCED LOGGING SETUP
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('vector_db_build_enhanced.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ENHANCED CONFIGURATION WITH MULTI-MODEL SUPPORT
class Config:
    """Enhanced configuration with multi-model and disease detection support"""
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # Source directories
    DATA_DIR = BASE_DIR / "data" / "final"
    DATASETS_DIR = BASE_DIR / "datasets"
    UPLOADS_DIR = BASE_DIR / "uploads"
    TRAINING_DIR = BASE_DIR / "training"
    
    # Target directory
    VECTOR_DB_DIR = BASE_DIR / "models" / "vector_db"
    
    # Species Classification Model configurations
    SPECIES_MODEL_CONFIGS = {
        'fish_species_model': {
            'path': BASE_DIR / "training" / "checkpoints" / "fish_model.pth",
            'class_mapping': BASE_DIR / "training" / "checkpoints" / "class_mapping1.json",
            'config': BASE_DIR / "training" / "configs" / "fish_config_improved.json",
            'priority': 1,
            'purpose': 'species_classification'
        },
        'best_model': {
            'path': BASE_DIR / "training" / "checkpoints" / "best_model.pth",
            'class_mapping': BASE_DIR / "training" / "checkpoints" / "class_mapping.json",
            'config': BASE_DIR / "training" / "configs" / "fish_config.json",
            'priority': 2,
            'purpose': 'species_classification'
        },
        'final_model': {
            'path': BASE_DIR / "training" / "checkpoints" / "final_model.pth",
            'class_mapping': BASE_DIR / "training" / "checkpoints" / "class_mapping.json",
            'config': BASE_DIR / "training" / "configs" / "fish_config.json",
            'priority': 3,
            'purpose': 'species_classification'
        }
    }
    
    # Disease Detection Model configurations
    DISEASE_MODEL_CONFIGS = {
        'disease_model_final': {
            'path': BASE_DIR / "training" / "checkpoints" / "best_efficientnet_freshwater.keras",
            'class_mapping': BASE_DIR / "training" / "checkpoints" / "disease_class_mapping.json",
            'priority': 1,
            'purpose': 'disease_detection'
        }
    }
    
    # Training report for additional metadata
    TRAINING_REPORT_PATH = BASE_DIR / "training" / "configs" / "training_report.json"
    
    # Embedding model
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Chunking parameters
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        'csv', 'json', 'pdf', 'txt',
        'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'
    }
    
    # Batch processing
    BATCH_SIZE = 50
    IMAGE_BATCH_SIZE = 32
    
    # File validation
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    MIN_FILE_SIZE = 1  # 1 byte
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Feature extraction settings
    EXTRACT_DEEP_FEATURES = True
    FEATURE_DIM = 1280
    
    # Ensemble prediction settings
    ENABLE_ENSEMBLE = True
    ENSEMBLE_THRESHOLD = 0.5
    
    @staticmethod
    def get_all_source_dirs() -> List[Path]:
        """Get all directories to scan for documents"""
        dirs = []
        for d in [Config.DATA_DIR, Config.DATASETS_DIR, Config.UPLOADS_DIR]:
            if d.exists():
                dirs.append(d)
        return dirs
    
    @staticmethod
    def is_path_safe(filepath: Path, allowed_base_dirs: List[Path]) -> bool:
        """Check if a file path is within allowed directories"""
        try:
            resolved_path = filepath.resolve()
            if '..' in filepath.parts:
                return False
            
            for base_dir in allowed_base_dirs:
                resolved_base = base_dir.resolve()
                try:
                    resolved_path.relative_to(resolved_base)
                    return True
                except ValueError:
                    continue
            return False
        except Exception as e:
            logger.error(f"Error checking path safety: {e}")
            return False
    
    @staticmethod
    def load_training_report() -> Optional[Dict]:
        """Load training report for enhanced metadata"""
        try:
            if Config.TRAINING_REPORT_PATH.exists():
                with open(Config.TRAINING_REPORT_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load training report: {e}")
        return None


# FISH DISEASE DETECTION SYSTEM
class FishDiseaseDetector:
    """Fish disease detection using Keras models"""
    
    def __init__(self):
        self.models = {}
        self.class_mappings = {}
        self.primary_model_name = None
        self.transforms = None
        
        logger.info("🏥 Initializing Fish Disease Detection System")
        self._load_all_models()
        self._setup_transforms()
    
    def _load_all_models(self):
        """Load all available disease detection models"""
        available_models = []
        
        for model_name, config in Config.DISEASE_MODEL_CONFIGS.items():
            if config['path'].exists():
                try:
                    model_info = self._load_single_model(
                        model_name,
                        str(config['path']),
                        str(config['class_mapping']) if config['class_mapping'].exists() else None
                    )
                    
                    if model_info:
                        available_models.append((config['priority'], model_name, model_info))
                        logger.info(f"✓ Loaded {model_name} (Priority: {config['priority']})")
                
                except Exception as e:
                    logger.error(f"✗ Failed to load {model_name}: {e}")
        
        if not available_models:
            logger.warning("⚠️ No disease detection models could be loaded!")
            return
        
        # Sort by priority
        available_models.sort(key=lambda x: x[0])
        self.primary_model_name = available_models[0][1]
        logger.info(f"🎯 Primary disease model: {self.primary_model_name}")
    
    def _load_single_model(
        self, 
        model_name: str,
        model_path: str, 
        class_mapping_path: Optional[str] = None
    ) -> Optional[Dict]:
        """Load a single Keras model"""
        try:
            # Load Keras model
            model = keras.models.load_model(model_path)
            self.models[model_name] = model
            
            # Load class mapping
            class_mapping = self._load_class_mapping(class_mapping_path)
            if not class_mapping:
                logger.warning(f"⚠️ No class mapping for {model_name}")
                return None
            
            self.class_mappings[model_name] = class_mapping
            
            return {
                'num_classes': len(class_mapping),
                'input_shape': model.input_shape
            }
            
        except Exception as e:
            logger.error(f"Error loading {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_class_mapping(self, class_mapping_path: Optional[str]) -> Optional[Dict]:
        """Load disease class mapping"""
        if not class_mapping_path or not Path(class_mapping_path).exists():
            return None
        
        try:
            with open(class_mapping_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
                
                # Handle different mapping formats
                if isinstance(mapping, dict):
                    if 'disease_to_id' in mapping:
                        return mapping['disease_to_id']
                    elif 'id_to_disease' in mapping:
                        id_to_disease = mapping['id_to_disease']
                        return {disease: int(idx) for idx, disease in id_to_disease.items()}
                    else:
                        return mapping
                
                return None
                
        except Exception as e:
            logger.error(f"Error loading class mapping: {e}")
            return None
    
    def _setup_transforms(self):
        """Setup image preprocessing for disease detection"""
        # Standard preprocessing for Keras models
        self.img_size = (224, 224)
    
    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Preprocess image for disease detection"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.img_size)
            img_array = np.array(img) / 255.0  # Normalize to [0, 1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            return img_array
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            return None
    
    def detect_disease(self, image_path: str) -> Dict[str, Any]:
        """Detect fish disease from image"""
        if not self.models:
            return {"status": "error", "message": "No disease models loaded"}
        
        try:
            # Preprocess image
            img_array = self.preprocess_image(image_path)
            if img_array is None:
                return {"status": "error", "message": "Image preprocessing failed"}
            
            # Use primary model for prediction
            model = self.models[self.primary_model_name]
            class_mapping = self.class_mappings[self.primary_model_name]
            
            # Predict
            predictions = model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Get disease name
            idx_to_class = {v: k for k, v in class_mapping.items()}
            disease_name = idx_to_class.get(predicted_class, f"Unknown_Disease_{predicted_class}")
            
            # Get top 3 predictions
            top3_indices = np.argsort(predictions[0])[-3:][::-1]
            top3_predictions = []
            for idx in top3_indices:
                disease = idx_to_class.get(idx, f"Unknown_Disease_{idx}")
                conf = float(predictions[0][idx])
                top3_predictions.append({
                    "disease": disease,
                    "confidence": conf
                })
            
            # Determine health status
            is_healthy = disease_name.lower() in ['healthy', 'normal', 'no_disease']
            
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
            logger.error(f"Disease detection error for {image_path}: {e}")
            return {"status": "error", "message": str(e)}
    
    @property
    def is_loaded(self) -> bool:
        """Check if any disease model is loaded"""
        return len(self.models) > 0


# ENHANCED MULTI-MODEL FISH CLASSIFICATION SYSTEM
class EnhancedFishClassifier:
    """Enhanced multi-model fish species classification with ensemble prediction"""
    
    def __init__(self, enable_ensemble: bool = True):
        self.device = Config.DEVICE
        self.models = {}  
        self.class_mappings = {}
        self.model_configs = {}  
        self.transforms = None
        self.enable_ensemble = enable_ensemble and Config.ENABLE_ENSEMBLE
        self.training_report = Config.load_training_report()
        
        logger.info(f"🐟 Initializing Fish Species Classification System on {self.device}")
        logger.info(f"🔀 Ensemble mode: {'ENABLED' if self.enable_ensemble else 'DISABLED'}")
        
        self._load_all_models()
        self._setup_transforms()
    
    def _load_all_models(self):
        """Load all available species classification models"""
        available_models = []
        
        for model_name, config in Config.SPECIES_MODEL_CONFIGS.items():
            if config['path'].exists():
                try:
                    model_info = self._load_single_model(
                        model_name,
                        str(config['path']),
                        str(config['class_mapping']) if config['class_mapping'].exists() else None,
                        str(config['config']) if config['config'].exists() else None
                    )
                    
                    if model_info:
                        available_models.append((config['priority'], model_name, model_info))
                        logger.info(f"✓ Loaded {model_name} (Priority: {config['priority']})")
                
                except Exception as e:
                    logger.error(f"✗ Failed to load {model_name}: {e}")
        
        if not available_models:
            logger.error("❌ No species models could be loaded!")
            return
        
        available_models.sort(key=lambda x: x[0])
        self.primary_model_name = available_models[0][1]
        logger.info(f"🎯 Primary species model: {self.primary_model_name}")
        
        if self.enable_ensemble and len(available_models) > 1:
            logger.info(f"🔀 Ensemble enabled with {len(available_models)} models")
    
    def _load_single_model(
        self, 
        model_name: str,
        model_path: str, 
        class_mapping_path: Optional[str] = None,
        config_path: Optional[str] = None
    ) -> Optional[Dict]:
        """Load a single PyTorch model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
                metadata = {
                    'epoch': checkpoint.get('epoch', 'unknown'),
                    'accuracy': checkpoint.get('accuracy', 'unknown'),
                    'loss': checkpoint.get('loss', 'unknown')
                }
            else:
                state_dict = checkpoint
                metadata = {}
            
            class_mapping = self._load_class_mapping(checkpoint, class_mapping_path)
            if not class_mapping:
                logger.warning(f"⚠️ No class mapping for {model_name}")
                return None
            
            num_classes = len(class_mapping)
            model_config = self._load_model_config(config_path) if config_path else {}
            
            model = self._create_model_architecture(num_classes)
            self._load_weights_safely(model, state_dict, model_name)
            
            model.to(self.device)
            model.eval()
            
            self.models[model_name] = model
            self.class_mappings[model_name] = class_mapping
            self.model_configs[model_name] = model_config
            
            return {
                'num_classes': num_classes,
                'metadata': metadata,
                'config': model_config
            }
            
        except Exception as e:
            logger.error(f"Error loading {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_class_mapping(self, checkpoint: Dict, external_path: Optional[str]) -> Optional[Dict[str, int]]:
        """Load class mapping from checkpoint or external file"""
        class_mapping = None
        
        if isinstance(checkpoint, dict):
            if 'species_mapping' in checkpoint:
                species_mapping = checkpoint['species_mapping']
                if isinstance(species_mapping, dict):
                    if 'species_to_id' in species_mapping:
                        class_mapping = species_mapping['species_to_id']
                    elif 'id_to_species' in species_mapping:
                        id_to_species = species_mapping['id_to_species']
                        class_mapping = {species: int(idx) for idx, species in id_to_species.items()}
                    else:
                        class_mapping = species_mapping
            elif 'class_mapping' in checkpoint:
                class_mapping = checkpoint['class_mapping']
            elif 'class_names' in checkpoint:
                class_mapping = {name: idx for idx, name in enumerate(checkpoint['class_names'])}
        
        if external_path and Path(external_path).exists():
            try:
                with open(external_path, 'r', encoding='utf-8') as f:
                    external_mapping = json.load(f)
                    
                    if isinstance(external_mapping, dict):
                        if 'species_to_id' in external_mapping:
                            class_mapping = external_mapping['species_to_id']
                        elif 'id_to_species' in external_mapping:
                            id_to_species = external_mapping['id_to_species']
                            class_mapping = {species: int(idx) for idx, species in id_to_species.items()}
                        else:
                            class_mapping = external_mapping
                        
                        logger.info(f"✓ Loaded external class mapping: {len(class_mapping)} classes")
            except Exception as e:
                logger.error(f"Error loading external class mapping: {e}")
        
        return class_mapping
    
    def _load_model_config(self, config_path: str) -> Dict:
        """Load model training configuration"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"⚠️ Could not load model config: {e}")
            return {}
    
    def _create_model_architecture(self, num_classes: int) -> nn.Module:
        """Create EfficientNet-B0 model"""
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        
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
    
    def _load_weights_safely(self, model: nn.Module, state_dict: Dict, model_name: str):
        """Load weights with automatic key remapping"""
        try:
            model.load_state_dict(state_dict, strict=True)
            logger.info(f"✓ {model_name}: Weights loaded (strict)")
        except RuntimeError:
            remapped = {}
            for key, value in state_dict.items():
                new_key = f'backbone.{key}' if not key.startswith('backbone.') else key
                remapped[new_key] = value
            
            try:
                model.load_state_dict(remapped, strict=False)
                logger.info(f"✓ {model_name}: Weights loaded (remapped, non-strict)")
            except:
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"✓ {model_name}: Weights loaded (non-strict)")
    
    def _setup_transforms(self):
        """Setup image preprocessing"""
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def classify_image(self, image_path: str) -> Dict[str, Any]:
        """Species classification"""
        if not self.models:
            return {"status": "error", "message": "No species models loaded"}
        
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transforms(img).unsqueeze(0).to(self.device)
            
            if not self.enable_ensemble or len(self.models) == 1:
                return self._single_model_predict(img_tensor, self.primary_model_name)
            
            return self._ensemble_predict(img_tensor)
            
        except Exception as e:
            logger.error(f"Classification error for {image_path}: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
    
    def _single_model_predict(self, img_tensor: torch.Tensor, model_name: str) -> Dict[str, Any]:
        """Single model prediction"""
        model = self.models[model_name]
        class_mapping = self.class_mappings[model_name]
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = predicted.item()
        confidence_score = confidence.item()
        
        idx_to_class = {v: k for k, v in class_mapping.items()}
        species_name = idx_to_class.get(predicted_class, f"Unknown_Class_{predicted_class}")
        
        top3_prob, top3_indices = torch.topk(probabilities, min(3, probabilities.size(1)))
        top3_predictions = []
        for prob, idx in zip(top3_prob[0], top3_indices[0]):
            class_name = idx_to_class.get(idx.item(), f"Unknown_Class_{idx.item()}")
            top3_predictions.append({
                "species": class_name,
                "confidence": prob.item()
            })
        
        return {
            "status": "success",
            "predicted_species": species_name,
            "confidence": confidence_score,
            "top3_predictions": top3_predictions,
            "model_used": model_name,
            "ensemble": False,
            "total_classes": len(class_mapping)
        }
    
    def _ensemble_predict(self, img_tensor: torch.Tensor) -> Dict[str, Any]:
        """Ensemble prediction from multiple models"""
        all_predictions = {}
        vote_counts = defaultdict(lambda: {'votes': 0, 'total_confidence': 0.0, 'models': []})
        
        for model_name, model in self.models.items():
            class_mapping = self.class_mappings[model_name]
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = predicted.item()
            confidence_score = confidence.item()
            
            idx_to_class = {v: k for k, v in class_mapping.items()}
            species_name = idx_to_class.get(predicted_class, f"Unknown_Class_{predicted_class}")
            
            all_predictions[model_name] = {
                'species': species_name,
                'confidence': confidence_score,
                'class_id': predicted_class
            }
            
            vote_counts[species_name]['votes'] += 1
            vote_counts[species_name]['total_confidence'] += confidence_score
            vote_counts[species_name]['models'].append(model_name)
        
        max_votes = max(vote_counts.values(), key=lambda x: (x['votes'], x['total_confidence']))
        consensus_species = [k for k, v in vote_counts.items() if v == max_votes][0]
        
        avg_confidence = max_votes['total_confidence'] / max_votes['votes']
        agreement = max_votes['votes'] / len(self.models)
        
        return {
            "status": "success",
            "predicted_species": consensus_species,
            "confidence": avg_confidence,
            "ensemble": True,
            "agreement": agreement,
            "models_agree": max_votes['votes'],
            "total_models": len(self.models),
            "model_predictions": all_predictions,
            "consensus_models": max_votes['models']
        }
    
    @property
    def is_loaded(self) -> bool:
        """Check if any model is loaded"""
        return len(self.models) > 0


# PRODUCTION DOCUMENT LOADER WITH ENHANCED ML
class ProductionDocumentLoader:
    """Enhanced document loader with species classification and disease detection"""
    
    def __init__(
        self, 
        fish_classifier: Optional[EnhancedFishClassifier] = None,
        disease_detector: Optional[FishDiseaseDetector] = None
    ):
        self.stats = {
            'pdf': 0, 'csv': 0, 'json': 0, 'txt': 0,
            'images': 0, 'classified_images': 0, 'ensemble_predictions': 0,
            'disease_detected_images': 0, 'healthy_fish': 0, 'diseased_fish': 0,
            'errors': 0, 'total_docs': 0, 'skipped_duplicates': 0,
            'security_rejections': 0
        }
        self.processed_files = set()
        self.error_log = []
        self.fish_classifier = fish_classifier
        self.disease_detector = disease_detector
        self.species_counts = defaultdict(int)
        self.disease_counts = defaultdict(int)
        
        self.allowed_dirs = Config.get_all_source_dirs()
    
    def validate_file(self, filepath: str) -> bool:
        """Enhanced file validation"""
        try:
            path = Path(filepath)
            
            if not path.exists():
                logger.warning(f"⚠️ File does not exist: {filepath}")
                return False
            
            file_size = path.stat().st_size
            if file_size > Config.MAX_FILE_SIZE:
                logger.warning(f"⚠️ File too large ({file_size} bytes): {filepath}")
                return False
            
            if file_size < Config.MIN_FILE_SIZE:
                logger.warning(f"⚠️ Empty file: {filepath}")
                return False
            
            ext = path.suffix.lower().lstrip('.')
            if ext not in Config.SUPPORTED_EXTENSIONS:
                return False
            
            if not Config.is_path_safe(path, self.allowed_dirs):
                logger.debug(f"🔒 Path outside allowed directories: {filepath}")
                self.stats['security_rejections'] += 1
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Validation error for {filepath}: {e}")
            return False
    
    def get_file_hash(self, filepath: str) -> str:
        """Generate file hash for deduplication"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"❌ Hash error for {filepath}: {e}")
            return ""
    
    def load_pdf(self, filepath: str) -> List[Document]:
        """Load PDF with enhanced metadata"""
        try:
            if not self.validate_file(filepath):
                return []
            
            file_hash = self.get_file_hash(filepath)
            if file_hash in self.processed_files:
                self.stats['skipped_duplicates'] += 1
                return []
            
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            
            if not docs:
                logger.warning(f"⚠️ No content from PDF: {filepath}")
                return []
            
            for i, doc in enumerate(docs):
                doc.metadata.update({
                    "source": filepath,
                    "filename": Path(filepath).name,
                    "type": "pdf",
                    "page": i + 1,
                    "total_pages": len(docs),
                    "file_hash": file_hash,
                    "file_size": Path(filepath).stat().st_size,
                    "loaded_at": datetime.now().isoformat()
                })
            
            self.processed_files.add(file_hash)
            self.stats['pdf'] += len(docs)
            logger.info(f"📄 PDF: {Path(filepath).name} ({len(docs)} pages)")
            return docs
            
        except Exception as e:
            self.stats['errors'] += 1
            self.error_log.append(f"PDF Error: {filepath} - {str(e)}")
            logger.error(f"❌ PDF error: {filepath} - {e}")
            return []
    
    def load_csv(self, filepath: str) -> List[Document]:
        """Load CSV files"""
        try:
            if not self.validate_file(filepath):
                return []
            
            file_hash = self.get_file_hash(filepath)
            if file_hash in self.processed_files:
                self.stats['skipped_duplicates'] += 1
                return []
            
            df = pd.read_csv(filepath)
            
            if df.empty:
                logger.warning(f"⚠️ Empty CSV: {filepath}")
                return []
            
            docs = []
            summary = f"CSV File: {Path(filepath).name}\nRows: {df.shape[0]} Columns: {df.shape[1]}"
            docs.append(Document(
                page_content=summary,
                metadata={
                    "source": filepath,
                    "filename": Path(filepath).name,
                    "type": "csv",
                    "file_hash": file_hash
                }
            ))
            
            self.processed_files.add(file_hash)
            self.stats['csv'] += 1
            logger.info(f"📊 CSV: {Path(filepath).name}")
            return docs
            
        except Exception as e:
            self.stats['errors'] += 1
            self.error_log.append(f"CSV Error: {filepath} - {str(e)}")
            return []
    
    def load_json(self, filepath: str) -> List[Document]:
        """Load JSON files"""
        try:
            if not self.validate_file(filepath):
                return []
            
            file_hash = self.get_file_hash(filepath)
            if file_hash in self.processed_files:
                self.stats['skipped_duplicates'] += 1
                return []
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            text = json.dumps(data, indent=2, ensure_ascii=False)
            doc = Document(
                page_content=text,
                metadata={
                    "source": filepath,
                    "filename": Path(filepath).name,
                    "type": "json",
                    "file_hash": file_hash
                }
            )
            
            self.processed_files.add(file_hash)
            self.stats['json'] += 1
            logger.info(f"📋 JSON: {Path(filepath).name}")
            return [doc]
            
        except Exception as e:
            self.stats['errors'] += 1
            self.error_log.append(f"JSON Error: {filepath} - {str(e)}")
            return []
    
    def load_txt(self, filepath: str) -> List[Document]:
        """Load text files"""
        try:
            if not self.validate_file(filepath):
                return []
            
            file_hash = self.get_file_hash(filepath)
            if file_hash in self.processed_files:
                self.stats['skipped_duplicates'] += 1
                return []
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                return []
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": filepath,
                    "filename": Path(filepath).name,
                    "type": "txt",
                    "file_hash": file_hash
                }
            )
            
            self.processed_files.add(file_hash)
            self.stats['txt'] += 1
            logger.info(f"📝 TXT: {Path(filepath).name}")
            return [doc]
            
        except Exception as e:
            self.stats['errors'] += 1
            self.error_log.append(f"TXT Error: {filepath} - {str(e)}")
            return []
    
    def load_image(self, filepath: str) -> List[Document]:
        """Enhanced image loading with species classification AND disease detection"""
        try:
            if not self.validate_file(filepath):
                return []
            
            file_hash = self.get_file_hash(filepath)
            if file_hash in self.processed_files:
                self.stats['skipped_duplicates'] += 1
                return []
            
            img = Image.open(filepath)
            
            # Species Classification
            species_result = None
            if self.fish_classifier and self.fish_classifier.is_loaded:
                species_result = self.fish_classifier.classify_image(filepath)
                if species_result['status'] == 'success':
                    self.stats['classified_images'] += 1
                    if species_result.get('ensemble', False):
                        self.stats['ensemble_predictions'] += 1
                    
                    species = species_result['predicted_species']
                    self.species_counts[species] += 1
            
            # Disease Detection
            disease_result = None
            if self.disease_detector and self.disease_detector.is_loaded:
                disease_result = self.disease_detector.detect_disease(filepath)
                if disease_result['status'] == 'success':
                    self.stats['disease_detected_images'] += 1
                    
                    disease = disease_result['predicted_disease']
                    self.disease_counts[disease] += 1
                    
                    if disease_result.get('is_healthy', False):
                        self.stats['healthy_fish'] += 1
                    else:
                        self.stats['diseased_fish'] += 1
            
            # Build content
            content_parts = [
                f"Image File: {Path(filepath).name}",
                f"Dimensions: {img.size[0]} × {img.size[1]} pixels",
                f"Format: {img.format}",
                f"Mode: {img.mode}",
                f"File Size: {Path(filepath).stat().st_size / 1024:.2f} KB"
            ]
            
            # Add species classification results
            if species_result and species_result['status'] == 'success':
                content_parts.extend([
                    f"\n🐟 Species Classification:",
                    f"Predicted Species: {species_result['predicted_species']}",
                    f"Confidence: {species_result['confidence']:.2%}"
                ])
                
                if species_result.get('ensemble'):
                    content_parts.append(
                        f"Model Agreement: {species_result['models_agree']}/{species_result['total_models']}"
                    )
            
            # Add disease detection results
            if disease_result and disease_result['status'] == 'success':
                health_emoji = "✅" if disease_result.get('is_healthy') else "⚠️"
                content_parts.extend([
                    f"\n{health_emoji} Disease Detection:",
                    f"Detected Condition: {disease_result['predicted_disease']}",
                    f"Confidence: {disease_result['confidence']:.2%}",
                    f"Health Status: {'Healthy' if disease_result.get('is_healthy') else 'Diseased'}"
                ])
                
                if disease_result.get('top3_predictions'):
                    content_parts.append("\nTop 3 Disease Predictions:")
                    for i, pred in enumerate(disease_result['top3_predictions'][:3], 1):
                        content_parts.append(f"{i}. {pred['disease']} ({pred['confidence']:.2%})")
            
            content_parts.append("\n[Image analyzed for fish species identification and disease detection]")
            content = "\n".join(content_parts)
            
            # Metadata
            metadata = {
                "source": filepath,
                "filename": Path(filepath).name,
                "type": "image",
                "width": img.size[0],
                "height": img.size[1],
                "format": img.format,
                "file_hash": file_hash,
                "loaded_at": datetime.now().isoformat()
            }
            
            # Add species metadata
            if species_result and species_result['status'] == 'success':
                metadata.update({
                    "ml_predicted_species": species_result['predicted_species'],
                    "ml_species_confidence": species_result['confidence'],
                    "ml_species_classified": True,
                    "ml_ensemble": species_result.get('ensemble', False)
                })
            
            # Add disease metadata
            if disease_result and disease_result['status'] == 'success':
                metadata.update({
                    "ml_predicted_disease": disease_result['predicted_disease'],
                    "ml_disease_confidence": disease_result['confidence'],
                    "ml_disease_detected": True,
                    "ml_is_healthy": disease_result.get('is_healthy', False),
                    "ml_health_status": "healthy" if disease_result.get('is_healthy') else "diseased"
                })
            
            doc = Document(page_content=content, metadata=metadata)
            
            self.processed_files.add(file_hash)
            self.stats['images'] += 1
            
            status_parts = []
            if species_result and species_result['status'] == 'success':
                status_parts.append(f"Species: {species_result['predicted_species']}")
            if disease_result and disease_result['status'] == 'success':
                status_parts.append(f"Disease: {disease_result['predicted_disease']}")
            
            status = " | ".join(status_parts) if status_parts else "Basic"
            logger.info(f"🖼️ Image ({status}): {Path(filepath).name}")
            
            return [doc]
            
        except Exception as e:
            self.stats['errors'] += 1
            self.error_log.append(f"Image Error: {filepath} - {str(e)}")
            logger.error(f"❌ Image error: {filepath} - {e}")
            return []
    
    def load_document(self, filepath: str) -> List[Document]:
        """Universal document loader"""
        ext = Path(filepath).suffix.lower().lstrip('.')
        
        loaders = {
            'pdf': self.load_pdf,
            'csv': self.load_csv,
            'json': self.load_json,
            'txt': self.load_txt,
            'jpg': self.load_image,
            'jpeg': self.load_image,
            'png': self.load_image,
            'gif': self.load_image,
            'bmp': self.load_image,
            'webp': self.load_image
        }
        
        return loaders.get(ext, lambda x: [])(filepath)


# ENHANCED PRODUCTION VECTOR DB BUILDER
class EnhancedVectorDBBuilder:
    """Production-grade vector database builder with species and disease detection"""
    
    def __init__(self, enable_ensemble: bool = True, enable_disease_detection: bool = True):
        logger.info("🚀 Initializing Enhanced Vector DB Builder...")
        
        # Initialize classifiers
        self.fish_classifier = EnhancedFishClassifier(enable_ensemble=enable_ensemble)
        self.disease_detector = FishDiseaseDetector() if enable_disease_detection else None
        
        self.loader = ProductionDocumentLoader(
            fish_classifier=self.fish_classifier,
            disease_detector=self.disease_detector
        )
        self.embeddings = None
        self.vector_db = None
        
        self.build_stats = {
            "start_time": None,
            "end_time": None,
            "total_files": 0,
            "total_documents": 0,
            "total_chunks": 0,
            "ml_classifications": 0,
            "ensemble_predictions": 0,
            "disease_detections": 0,
            "healthy_fish_count": 0,
            "diseased_fish_count": 0,
            "skipped_duplicates": 0,
            "security_rejections": 0,
            "duration_seconds": 0,
            "species_models_loaded": len(self.fish_classifier.models),
            "disease_models_loaded": len(self.disease_detector.models) if self.disease_detector else 0,
            "primary_species_model": self.fish_classifier.primary_model_name if self.fish_classifier.models else None,
            "primary_disease_model": self.disease_detector.primary_model_name if self.disease_detector and self.disease_detector.is_loaded else None
        }
        
        logger.info(f"✓ Species Models: {self.build_stats['species_models_loaded']}")
        logger.info(f"✓ Disease Models: {self.build_stats['disease_models_loaded']}")
    
    def scan_directories(self) -> List[str]:
        """Scan all source directories"""
        all_files = []
        
        for directory in Config.get_all_source_dirs():
            logger.info(f"📂 Scanning: {directory}")
            
            for root, _, files in os.walk(str(directory)):
                for file in files:
                    filepath = Path(root) / file
                    ext = filepath.suffix.lower().lstrip('.')
                    if ext in Config.SUPPORTED_EXTENSIONS:
                        all_files.append(str(filepath))
        
        logger.info(f"📁 Found {len(all_files)} files")
        return all_files
    
    def load_all_documents(self, filepaths: List[str]) -> List[Document]:
        """Load all documents with ML classification and disease detection"""
        all_docs = []
        
        logger.info("📥 Loading documents with species classification and disease detection...")
        for filepath in tqdm(filepaths, desc="Processing files"):
            docs = self.loader.load_document(filepath)
            all_docs.extend(docs)
            self.loader.stats['total_docs'] += len(docs)
        
        logger.info(f"✓ Loaded {len(all_docs)} documents")
        return all_docs
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into optimized chunks"""
        logger.info("✂️ Chunking documents...")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = splitter.split_documents(documents)
        
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)
        
        logger.info(f"✓ Created {len(chunks)} chunks")
        return chunks
    
    def build_vector_db(self, chunks: List[Document]) -> Chroma:
        """Build vector database with optimized embedding"""
        logger.info("🔮 Initializing embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
       )

        
        logger.info("🏗️ Building vector database...")
        
        Config.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        
        if len(chunks) > Config.BATCH_SIZE:
            logger.info(f"Processing in batches of {Config.BATCH_SIZE}...")
            
            self.vector_db = Chroma.from_documents(
                chunks[:Config.BATCH_SIZE],
                embedding=self.embeddings,
                persist_directory=str(Config.VECTOR_DB_DIR),
                collection_name="meenasetu_enhanced"
            )
            
            for i in tqdm(range(Config.BATCH_SIZE, len(chunks), Config.BATCH_SIZE),
                         desc="Building vector DB"):
                batch = chunks[i:i + Config.BATCH_SIZE]
                self.vector_db.add_documents(batch)
        else:
            self.vector_db = Chroma.from_documents(
                chunks,
                embedding=self.embeddings,
                persist_directory=str(Config.VECTOR_DB_DIR),
                collection_name="meenasetu_enhanced"
            )
        
        logger.info("✓ Vector database built successfully")
        return self.vector_db
    
    def save_enhanced_report(self):
        """Save comprehensive build report"""
        report = {
            "build_info": {
                "timestamp": datetime.now().isoformat(),
                "version": "3.0-species-and-disease",
                "stats": self.build_stats
            },
            "loader_stats": self.loader.stats,
            "species_distribution": dict(self.loader.species_counts),
            "disease_distribution": dict(self.loader.disease_counts),
            "ml_integration": {
                "species_models_loaded": len(self.fish_classifier.models),
                "disease_models_loaded": len(self.disease_detector.models) if self.disease_detector else 0,
                "primary_species_model": self.fish_classifier.primary_model_name if self.fish_classifier.models else None,
                "primary_disease_model": self.disease_detector.primary_model_name if self.disease_detector and self.disease_detector.is_loaded else None,
                "ensemble_enabled": self.fish_classifier.enable_ensemble,
                "device": str(Config.DEVICE),
                "images_classified": self.loader.stats['classified_images'],
                "ensemble_predictions": self.loader.stats['ensemble_predictions'],
                "disease_detections": self.loader.stats['disease_detected_images'],
                "healthy_fish": self.loader.stats['healthy_fish'],
                "diseased_fish": self.loader.stats['diseased_fish']
            },
            "errors": self.loader.error_log,
            "config": {
                "chunk_size": Config.CHUNK_SIZE,
                "chunk_overlap": Config.CHUNK_OVERLAP,
                "embed_model": Config.EMBED_MODEL,
                "batch_size": Config.BATCH_SIZE,
                "source_directories": [str(d) for d in Config.get_all_source_dirs()],
                "supported_extensions": list(Config.SUPPORTED_EXTENSIONS)
            }
        }
        
        report_path = Config.BASE_DIR / "vector_db_enhanced_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save species distribution
        if self.loader.species_counts:
            species_path = Config.BASE_DIR / "species_distribution.json"
            with open(species_path, 'w', encoding='utf-8') as f:
                json.dump(dict(self.loader.species_counts), f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Species distribution saved: {species_path}")
        
        # Save disease distribution
        if self.loader.disease_counts:
            disease_path = Config.BASE_DIR / "disease_distribution.json"
            with open(disease_path, 'w', encoding='utf-8') as f:
                json.dump(dict(self.loader.disease_counts), f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Disease distribution saved: {disease_path}")
        
        logger.info(f"💾 Enhanced report saved: {report_path}")
    
    def build(self):
        """Main enhanced build pipeline"""
        logger.info("=" * 80)
        logger.info("🚀 ENHANCED VECTOR DATABASE BUILD WITH DISEASE DETECTION")
        logger.info("=" * 80)
        
        self.build_stats["start_time"] = datetime.now().isoformat()
        start = datetime.now()
        
        # Step 1: Scan
        filepaths = self.scan_directories()
        self.build_stats["total_files"] = len(filepaths)
        
        if not filepaths:
            logger.warning("⚠️ No files found!")
            return
        
        # Step 2: Load with ML
        documents = self.load_all_documents(filepaths)
        
        if not documents:
            logger.warning("⚠️ No documents loaded!")
            return
        
        self.build_stats["total_documents"] = len(documents)
        
        # Step 3: Chunk
        chunks = self.chunk_documents(documents)
        self.build_stats["total_chunks"] = len(chunks)
        self.build_stats["ml_classifications"] = self.loader.stats['classified_images']
        self.build_stats["ensemble_predictions"] = self.loader.stats['ensemble_predictions']
        self.build_stats["disease_detections"] = self.loader.stats['disease_detected_images']
        self.build_stats["healthy_fish_count"] = self.loader.stats['healthy_fish']
        self.build_stats["diseased_fish_count"] = self.loader.stats['diseased_fish']
        self.build_stats["skipped_duplicates"] = self.loader.stats['skipped_duplicates']
        self.build_stats["security_rejections"] = self.loader.stats['security_rejections']
        
        # Step 4: Build vector DB
        self.build_vector_db(chunks)
        
        # Step 5: Finalize
        self.build_stats["end_time"] = datetime.now().isoformat()
        self.build_stats["duration_seconds"] = (datetime.now() - start).total_seconds()
        
        # Step 6: Save reports
        self.save_enhanced_report()
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print comprehensive build summary"""
        logger.info("\n" + "=" * 80)
        logger.info("✨ ENHANCED BUILD COMPLETE ✨")
        logger.info("=" * 80)
        logger.info(f"📁 Total Files Found: {self.build_stats['total_files']}")
        logger.info(f"📄 Total Documents: {self.build_stats['total_documents']}")
        logger.info(f"✂️ Total Chunks: {self.build_stats['total_chunks']}")
        logger.info(f"⏭️ Skipped Duplicates: {self.build_stats['skipped_duplicates']}")
        logger.info(f"🔒 Security Rejections: {self.build_stats['security_rejections']}")
        logger.info(f"⏱️ Duration: {self.build_stats['duration_seconds']:.2f}s")
        
        logger.info(f"\n📊 File Type Breakdown:")
        logger.info(f"   PDFs: {self.loader.stats['pdf']}")
        logger.info(f"   CSVs: {self.loader.stats['csv']}")
        logger.info(f"   JSONs: {self.loader.stats['json']}")
        logger.info(f"   TXTs: {self.loader.stats['txt']}")
        logger.info(f"   Images: {self.loader.stats['images']}")
        logger.info(f"   Errors: {self.loader.stats['errors']}")
        
        if self.fish_classifier.is_loaded:
            logger.info(f"\n🐟 Species Classification Summary:")
            logger.info(f"   Models Loaded: {len(self.fish_classifier.models)}")
            logger.info(f"   Primary Model: {self.fish_classifier.primary_model_name}")
            logger.info(f"   Images Classified: {self.loader.stats['classified_images']}")
            logger.info(f"   Ensemble Predictions: {self.loader.stats['ensemble_predictions']}")
            
            if self.loader.species_counts:
                logger.info(f"\n   Top 5 Detected Species:")
                top_species = sorted(
                    self.loader.species_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                for species, count in top_species:
                    logger.info(f"   • {species}: {count} images")
        
        if self.disease_detector and self.disease_detector.is_loaded:
            logger.info(f"\n🏥 Disease Detection Summary:")
            logger.info(f"   Models Loaded: {len(self.disease_detector.models)}")
            logger.info(f"   Primary Model: {self.disease_detector.primary_model_name}")
            logger.info(f"   Disease Detections: {self.loader.stats['disease_detected_images']}")
            logger.info(f"   Healthy Fish: {self.loader.stats['healthy_fish']}")
            logger.info(f"   Diseased Fish: {self.loader.stats['diseased_fish']}")
            
            if self.loader.disease_counts:
                logger.info(f"\n   Top 5 Detected Diseases:")
                top_diseases = sorted(
                    self.loader.disease_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                for disease, count in top_diseases:
                    logger.info(f"   • {disease}: {count} images")
        
        if self.loader.error_log:
            logger.info(f"\n⚠️ Errors: {len(self.loader.error_log)}")
            logger.info(f"   Check 'vector_db_enhanced_report.json' for details")
        
        logger.info("=" * 80 + "\n")


# MAIN EXECUTION
def main():
    """Main execution with species classification and disease detection"""
    try:
        # Build with both species classification and disease detection enabled
        builder = EnhancedVectorDBBuilder(
            enable_ensemble=True,
            enable_disease_detection=True
        )
        builder.build()
        
        logger.info("🎉 Enhanced vector database build completed!")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Build interrupted by user")
    
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()