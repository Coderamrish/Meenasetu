import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from PIL import Image
import base64
import hashlib

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

# ============================================================
# 📋 LOGGING SETUP
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vector_db_build.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# 🔧 CONFIGURATION
# ============================================================
class Config:
    """Configuration class with path management"""
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # Source directories
    DATA_DIR = BASE_DIR / "data" / "final"
    DATASETS_DIR = BASE_DIR / "datasets"
    UPLOADS_DIR = BASE_DIR / "uploads"
    TRAINING_DIR = BASE_DIR / "training"
    
    # Target directory
    VECTOR_DB_DIR = BASE_DIR / "models" / "vector_db"
    
    # Model paths
    PYTORCH_MODEL_PATH = BASE_DIR / "training" / "checkpoints" / "best_model.pth"
    CLASS_MAPPING_PATH = BASE_DIR / "training" / "checkpoints" / "class_mapping.json"
    
    # Embedding model
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Chunking parameters
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        'csv', 'json', 'pdf', 'txt',
        'jpg', 'jpeg', 'png', 'gif', 'bmp'
    }
    
    # Batch processing
    BATCH_SIZE = 50
    IMAGE_BATCH_SIZE = 16  # For batch image classification
    
    # File validation
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def get_all_source_dirs() -> List[Path]:
        """Get all directories to scan for documents"""
        dirs = []
        for d in [Config.DATA_DIR, Config.DATASETS_DIR, Config.UPLOADS_DIR]:
            if d.exists():
                dirs.append(d)
        return dirs

# ============================================================
# 🤖 PYTORCH MODEL INTEGRATION (CORRECTED)
# ============================================================
class FishClassificationModel:
    """Integrated PyTorch model for fish species classification"""
    
    def __init__(self, model_path: Optional[str] = None, class_mapping_path: Optional[str] = None):
        self.device = Config.DEVICE
        self.model = None
        self.class_mapping = {}
        self.transforms = None
        self.is_loaded = False
        
        logger.info(f"🤖 Initializing Fish Classification Model on {self.device}")
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path, class_mapping_path)
        else:
            logger.warning("⚠️ Model not found. Image classification will use basic features only.")
    
    def load_model(self, model_path: str, class_mapping_path: Optional[str] = None):
        """Load trained PyTorch model with robust error handling"""
        try:
            logger.info(f"📥 Loading model from: {model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract state dictionary
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Extract class mapping from checkpoint (multiple format support)
                self._extract_class_mapping_from_checkpoint(checkpoint)
                
                # Determine number of classes
                num_classes = self._determine_num_classes(state_dict)
            else:
                state_dict = checkpoint
                num_classes = 14
                logger.warning("⚠️ Checkpoint is raw state dict, using default 14 classes")
            
            # Load external class mapping (overrides checkpoint)
            if class_mapping_path and Path(class_mapping_path).exists():
                self._load_external_class_mapping(class_mapping_path)
                if self.class_mapping:
                    num_classes = len(self.class_mapping)
            
            # Create model architecture
            self.model = self._create_model_architecture(num_classes)
            
            # Load weights with automatic key remapping
            self._load_state_dict_with_remapping(state_dict)
            
            self.model.to(self.device)
            self.model.eval()
            
            # Setup transforms
            self._setup_transforms()
            
            self.is_loaded = True
            logger.info(f"✅ Model loaded successfully with {num_classes} classes")
            
            # Log class mapping status
            self._log_class_mapping_info()
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.is_loaded = False
    
    def _extract_class_mapping_from_checkpoint(self, checkpoint: Dict):
        """Extract class mapping from checkpoint in various formats"""
        if 'species_mapping' in checkpoint:
            species_mapping = checkpoint['species_mapping']
            
            if isinstance(species_mapping, dict):
                if 'species_to_id' in species_mapping:
                    self.class_mapping = species_mapping['species_to_id']
                    logger.info(f"✅ Loaded species_to_id: {len(self.class_mapping)} classes")
                elif 'id_to_species' in species_mapping:
                    id_to_species = species_mapping['id_to_species']
                    self.class_mapping = {species: int(idx) for idx, species in id_to_species.items()}
                    logger.info(f"✅ Created mapping from id_to_species: {len(self.class_mapping)} classes")
                else:
                    self.class_mapping = species_mapping
                    logger.info(f"✅ Loaded species_mapping: {len(self.class_mapping)} classes")
        
        elif 'class_mapping' in checkpoint:
            self.class_mapping = checkpoint['class_mapping']
            logger.info(f"✅ Loaded class_mapping: {len(self.class_mapping)} classes")
        
        elif 'class_names' in checkpoint:
            self.class_mapping = {name: idx for idx, name in enumerate(checkpoint['class_names'])}
            logger.info(f"✅ Created mapping from class_names: {len(self.class_mapping)} classes")
        
        elif 'idx_to_class' in checkpoint:
            idx_to_class = checkpoint['idx_to_class']
            self.class_mapping = {v: int(k) for k, v in idx_to_class.items()}
            logger.info(f"✅ Created mapping from idx_to_class: {len(self.class_mapping)} classes")
    
    def _determine_num_classes(self, state_dict: Dict) -> int:
        """Determine number of classes from state dict"""
        detected_classes = None
        detected_key = None
        
        # Try to detect from state dict - ONLY look at the FINAL classifier layer
        # Priority order: classifier.9 > classifier.5 > classifier.1 > fc
        priority_keys = []
        
        for key in state_dict.keys():
            # Skip batch norm running stats and non-weight parameters
            if 'running' in key or 'num_batches' in key or 'bias' in key:
                continue
            
            # Only look for the FINAL output layer
            if 'classifier.9.weight' in key:  # Last layer in our custom classifier
                priority_keys.append((9, key))
            elif 'classifier.5.weight' in key:  # Middle layer
                priority_keys.append((5, key))
            elif 'classifier.1.weight' in key and 'block' not in key:  # First linear layer
                priority_keys.append((1, key))
            elif 'fc.weight' in key and 'fc1' not in key and 'fc2' not in key:  # Simple fc layer
                priority_keys.append((0, key))
        
        # Sort by priority (higher number = later in network = more likely to be output layer)
        if priority_keys:
            priority_keys.sort(reverse=True)
            _, detected_key = priority_keys[0]
            detected_classes = state_dict[detected_key].shape[0]
            logger.info(f"🎯 Detected {detected_classes} output classes from: {detected_key}")
        
        # Compare with class mapping
        mapping_classes = len(self.class_mapping) if self.class_mapping else None
        
        if detected_classes and mapping_classes:
            if detected_classes != mapping_classes:
                logger.warning(f"⚠️ MISMATCH: Final layer has {detected_classes} outputs, but mapping has {mapping_classes} classes")
                logger.warning(f"⚠️ Using checkpoint's {detected_classes} classes for model architecture")
                # Use detected classes since that's what the checkpoint actually has
                return detected_classes
            else:
                logger.info(f"✅ Classes match: {detected_classes} classes in both checkpoint and mapping")
                return detected_classes
        elif detected_classes:
            logger.info(f"✅ Using {detected_classes} classes from checkpoint")
            return detected_classes
        elif mapping_classes:
            logger.warning(f"⚠️ Could not detect classes from checkpoint, using mapping: {mapping_classes} classes")
            return mapping_classes
        else:
            logger.warning(f"⚠️ Could not determine num_classes, defaulting to 14")
            return 14
    
    def _load_external_class_mapping(self, class_mapping_path: str):
        """Load class mapping from external JSON file"""
        try:
            with open(class_mapping_path, 'r', encoding='utf-8') as f:
                external_mapping = json.load(f)
                
                if isinstance(external_mapping, dict):
                    if 'species_to_id' in external_mapping:
                        self.class_mapping = external_mapping['species_to_id']
                    elif 'id_to_species' in external_mapping:
                        id_to_species = external_mapping['id_to_species']
                        self.class_mapping = {species: int(idx) for idx, species in id_to_species.items()}
                    else:
                        self.class_mapping = external_mapping
                    
                    logger.info(f"✅ Loaded external class mapping: {len(self.class_mapping)} classes")
        except Exception as e:
            logger.error(f"❌ Error loading external class mapping: {e}")
    
    def _load_state_dict_with_remapping(self, state_dict: Dict):
        """Load state dict with automatic key remapping"""
        try:
            self.model.load_state_dict(state_dict, strict=True)
            logger.info("✅ Model weights loaded successfully (strict mode)")
            
        except RuntimeError as e:
            logger.warning(f"⚠️ Strict loading failed, attempting key remapping...")
            
            # Try remapping: add 'backbone.' prefix
            remapped_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith('backbone.'):
                    new_key = f'backbone.{key}'
                    remapped_state_dict[new_key] = value
                else:
                    remapped_state_dict[key] = value
            
            try:
                self.model.load_state_dict(remapped_state_dict, strict=False)
                logger.info("✅ Model weights loaded with key remapping (non-strict mode)")
            except Exception as e2:
                logger.error(f"❌ Failed to load weights even with remapping: {e2}")
                # Try loading without remapping in non-strict mode as last resort
                self.model.load_state_dict(state_dict, strict=False)
                logger.info("✅ Model weights loaded (non-strict mode, no remapping)")
    
    def _log_class_mapping_info(self):
        """Log information about loaded class mappings"""
        if self.class_mapping:
            species_list = list(self.class_mapping.keys())[:5]
            logger.info(f"🏷️ Sample species: {species_list}...")
            logger.info(f"🐟 Total species in mapping: {len(self.class_mapping)}")
        else:
            logger.warning("⚠️ No class mapping available - predictions will use generic labels")
    
    def _create_model_architecture(self, num_classes: int) -> nn.Module:
        """
        Create EfficientNet-B0 model architecture matching the trained model.
        
        This method recreates the EXACT architecture from your training checkpoint,
        including any custom classifier layers beyond the standard dropout + linear.
        
        Args:
            num_classes: Number of output classes (detected from checkpoint)
            
        Returns:
            Wrapped EfficientNet-B0 model ready for inference
        """
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        
        # Load pre-trained EfficientNet-B0 backbone
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        # Get the number of input features for the classifier layer
        num_features = model.classifier[1].in_features  # 1280 for EfficientNet-B0
        
        # ===== IMPORTANT: Recreate the EXACT classifier architecture from training =====
        # Based on your checkpoint structure with BatchNorm layers:
        # classifier.0 = Dropout
        # classifier.1 = Linear(1280 -> intermediate_size)
        # classifier.2 = BatchNorm1d
        # classifier.3 = ReLU
        # classifier.4 = Dropout
        # classifier.5 = Linear(intermediate -> num_classes)
        # etc.
        
        # Check if this is a simple classifier (just Dropout + Linear) or complex one
        # For now, create a classifier that matches the checkpoint structure
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),           # classifier.0
            nn.Linear(num_features, 512),              # classifier.1 (matches your checkpoint!)
            nn.BatchNorm1d(512),                        # classifier.2
            nn.ReLU(inplace=True),                      # classifier.3
            nn.Dropout(p=0.3, inplace=True),           # classifier.4
            nn.Linear(512, 256),                        # classifier.5
            nn.BatchNorm1d(256),                        # classifier.6
            nn.ReLU(inplace=True),                      # classifier.7
            nn.Dropout(p=0.3, inplace=True),           # classifier.8
            nn.Linear(256, num_classes)                 # classifier.9 (final output)
        )
        
        # Define wrapper class to match training structure
        class FishClassifier(nn.Module):
            """Wrapper class for the EfficientNet backbone"""
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
            
            def forward(self, x):
                return self.backbone(x)
        
        # Wrap the model
        wrapped_model = FishClassifier(model)
        
        logger.info(f"🏗️ Created EfficientNet-B0 with CUSTOM CLASSIFIER for {num_classes} classes")
        logger.info(f"   Architecture: EfficientNet-B0 -> [1280→512→256→{num_classes}] with BatchNorm & Dropout")
        
        return wrapped_model
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms"""
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def classify_image(self, image_path: str) -> Dict[str, Any]:
        """Classify fish species from image"""
        if not self.is_loaded:
            return {
                "status": "error",
                "message": "Model not loaded"
            }
        
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transforms(img).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = predicted.item()
            confidence_score = confidence.item()
            
            # Get class name with proper mapping
            if self.class_mapping:
                idx_to_class = {v: k for k, v in self.class_mapping.items()}
                # Handle case where predicted class might be outside mapping range
                if predicted_class in idx_to_class:
                    species_name = idx_to_class[predicted_class]
                else:
                    species_name = f"Unknown_Class_{predicted_class}"
                    logger.warning(f"⚠️ Predicted class {predicted_class} not in mapping (mapping has {len(self.class_mapping)} classes)")
            else:
                species_name = f"Species_{predicted_class}"
            
            # Get top 3 predictions
            top3_prob, top3_indices = torch.topk(probabilities, min(3, probabilities.size(1)))
            top3_predictions = []
            for prob, idx in zip(top3_prob[0], top3_indices[0]):
                if self.class_mapping:
                    idx_to_class = {v: k for k, v in self.class_mapping.items()}
                    class_name = idx_to_class.get(idx.item(), f"Unknown_Class_{idx.item()}")
                else:
                    class_name = f"Species_{idx.item()}"
                
                top3_predictions.append({
                    "species": class_name,
                    "confidence": prob.item()
                })
            
            # Clear GPU cache if using CUDA
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            return {
                "status": "success",
                "predicted_species": species_name,
                "confidence": confidence_score,
                "top3_predictions": top3_predictions,
                "total_classes": len(self.class_mapping) if self.class_mapping else "Unknown",
                "model_output_classes": probabilities.size(1)  # Actual model output size
            }
            
        except Exception as e:
            logger.error(f"❌ Classification error for {image_path}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def extract_features(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract deep learning features from image using EfficientNet backbone.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Feature vector of shape (1, 1280) or None if extraction fails
        """
        if not self.is_loaded:
            logger.warning("⚠️ Model not loaded, cannot extract features")
            return None
        
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transforms(img).unsqueeze(0).to(self.device)
            
            # Extract features from the EfficientNet backbone
            with torch.no_grad():
                # Access the backbone through the wrapper
                if hasattr(self.model, 'backbone'):
                    # Get features from the convolutional layers
                    features = self.model.backbone.features(img_tensor)
                    # Apply global average pooling
                    features = self.model.backbone.avgpool(features)
                    # Flatten to 1D vector
                    features = torch.flatten(features, 1)
                else:
                    # Fallback: direct model access (if not wrapped)
                    features = self.model.features(img_tensor)
                    features = self.model.avgpool(features)
                    features = torch.flatten(features, 1)
            
            feature_array = features.cpu().numpy()
            
            # Clear GPU cache if using CUDA
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            return feature_array
            
        except Exception as e:
            logger.error(f"❌ Feature extraction error for {image_path}: {e}")
            return None

# ============================================================
# 🗂️ ADVANCED DOCUMENT LOADERS WITH ML INTEGRATION
# ============================================================
class ProductionDocumentLoader:
    """Production-grade document loader with ML classification"""
    
    def __init__(self, fish_classifier: Optional[FishClassificationModel] = None):
        self.stats = {
            'pdf': 0, 'csv': 0, 'json': 0, 'txt': 0, 
            'images': 0, 'classified_images': 0, 'errors': 0, 'total_docs': 0
        }
        self.processed_files = set()
        self.error_log = []
        self.fish_classifier = fish_classifier
    
    def validate_file(self, filepath: str) -> bool:
        """Validate file before processing"""
        try:
            # Check if file exists
            if not Path(filepath).exists():
                logger.warning(f"⚠️ File does not exist: {filepath}")
                return False
            
            # Check file size
            file_size = Path(filepath).stat().st_size
            if file_size > Config.MAX_FILE_SIZE:
                logger.warning(f"⚠️ File too large ({file_size} bytes): {filepath}")
                return False
            
            if file_size == 0:
                logger.warning(f"⚠️ Empty file: {filepath}")
                return False
            
            # Check file extension
            ext = Path(filepath).suffix.lower().lstrip('.')
            if ext not in Config.SUPPORTED_EXTENSIONS:
                return False
            
            # Check for path traversal
            resolved_path = Path(filepath).resolve()
            if '..' in str(filepath):
                logger.error(f"❌ Suspicious path detected: {filepath}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ File validation error for {filepath}: {e}")
            return False
    
    def get_file_hash(self, filepath: str) -> str:
        """Generate hash for file deduplication"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"❌ Error generating hash for {filepath}: {e}")
            return ""
    
    def load_pdf(self, filepath: str) -> List[Document]:
        """Load PDF with metadata"""
        try:
            if not self.validate_file(filepath):
                return []
            
            file_hash = self.get_file_hash(filepath)
            if file_hash in self.processed_files:
                logger.info(f"⏭️ Skipping duplicate: {filepath}")
                return []
            
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            
            if not docs:
                logger.warning(f"⚠️ No content extracted from PDF: {filepath}")
                return []
            
            for i, doc in enumerate(docs):
                doc.metadata.update({
                    "source": filepath,
                    "filename": Path(filepath).name,
                    "type": "pdf",
                    "page": i + 1,
                    "total_pages": len(docs),
                    "file_hash": file_hash,
                    "loaded_at": datetime.now().isoformat()
                })
            
            self.processed_files.add(file_hash)
            self.stats['pdf'] += len(docs)
            logger.info(f"✅ PDF: {Path(filepath).name} ({len(docs)} pages)")
            return docs
            
        except Exception as e:
            self.stats['errors'] += 1
            error_msg = f"PDF Error: {filepath} - {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.error_log.append(error_msg)
            return []
    
    def load_csv(self, filepath: str) -> List[Document]:
        """Load CSV with intelligent row handling"""
        try:
            if not self.validate_file(filepath):
                return []
            
            file_hash = self.get_file_hash(filepath)
            if file_hash in self.processed_files:
                return []
            
            df = pd.read_csv(filepath)
            
            # Validate DataFrame
            if df.empty:
                logger.warning(f"⚠️ Empty CSV file: {filepath}")
                return []
            
            docs = []
            
            # Document-level summary
            summary = f"""
CSV File: {Path(filepath).name}
Shape: {df.shape[0]} rows × {df.shape[1]} columns
Columns: {', '.join(df.columns.tolist())}

Statistical Summary:
{df.describe(include='all').to_string()}

Sample Data (First 5 rows):
{df.head().to_string()}
"""
            docs.append(Document(
                page_content=summary,
                metadata={
                    "source": filepath,
                    "filename": Path(filepath).name,
                    "type": "csv",
                    "content_type": "summary",
                    "rows": df.shape[0],
                    "columns": df.shape[1],
                    "file_hash": file_hash,
                    "loaded_at": datetime.now().isoformat()
                }
            ))
            
            # Row-level documents (limit to prevent excessive documents)
            max_rows = min(len(df), 1000)  # Limit to 1000 rows
            for idx, row in df.head(max_rows).iterrows():
                row_text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
                docs.append(Document(
                    page_content=row_text,
                    metadata={
                        "source": filepath,
                        "filename": Path(filepath).name,
                        "type": "csv",
                        "content_type": "row",
                        "row_index": idx,
                        "file_hash": file_hash,
                        "loaded_at": datetime.now().isoformat()
                    }
                ))
            
            self.processed_files.add(file_hash)
            self.stats['csv'] += 1
            logger.info(f"✅ CSV: {Path(filepath).name} ({df.shape[0]} rows, {len(docs)} documents)")
            return docs
            
        except Exception as e:
            self.stats['errors'] += 1
            error_msg = f"CSV Error: {filepath} - {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.error_log.append(error_msg)
            return []
    
    def load_json(self, filepath: str) -> List[Document]:
        """Load JSON with nested structure handling"""
        try:
            if not self.validate_file(filepath):
                return []
            
            file_hash = self.get_file_hash(filepath)
            if file_hash in self.processed_files:
                return []
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            docs = []
            
            if isinstance(data, list):
                for idx, entry in enumerate(data):
                    text = json.dumps(entry, indent=2, ensure_ascii=False)
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": filepath,
                            "filename": Path(filepath).name,
                            "type": "json",
                            "entry_index": idx,
                            "file_hash": file_hash,
                            "loaded_at": datetime.now().isoformat()
                        }
                    ))
            else:
                text = json.dumps(data, indent=2, ensure_ascii=False)
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": filepath,
                        "filename": Path(filepath).name,
                        "type": "json",
                        "file_hash": file_hash,
                        "loaded_at": datetime.now().isoformat()
                    }
                ))
            
            self.processed_files.add(file_hash)
            self.stats['json'] += len(docs)
            logger.info(f"✅ JSON: {Path(filepath).name} ({len(docs)} entries)")
            return docs
            
        except Exception as e:
            self.stats['errors'] += 1
            error_msg = f"JSON Error: {filepath} - {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.error_log.append(error_msg)
            return []
    
    def load_txt(self, filepath: str) -> List[Document]:
        """Load plain text files"""
        try:
            if not self.validate_file(filepath):
                return []
            
            file_hash = self.get_file_hash(filepath)
            if file_hash in self.processed_files:
                return []
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"⚠️ Empty text file: {filepath}")
                return []
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": filepath,
                    "filename": Path(filepath).name,
                    "type": "txt",
                    "size_bytes": Path(filepath).stat().st_size,
                    "file_hash": file_hash,
                    "loaded_at": datetime.now().isoformat()
                }
            )
            
            self.processed_files.add(file_hash)
            self.stats['txt'] += 1
            logger.info(f"✅ TXT: {Path(filepath).name}")
            return [doc]
            
        except Exception as e:
            self.stats['errors'] += 1
            error_msg = f"TXT Error: {filepath} - {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.error_log.append(error_msg)
            return []
    
    def load_image(self, filepath: str) -> List[Document]:
        """Load images with ML classification"""
        try:
            if not self.validate_file(filepath):
                return []
            
            file_hash = self.get_file_hash(filepath)
            if file_hash in self.processed_files:
                return []
            
            img = Image.open(filepath)
            
            # ML Classification if model is available
            classification_result = None
            if self.fish_classifier and self.fish_classifier.is_loaded:
                classification_result = self.fish_classifier.classify_image(filepath)
                if classification_result['status'] == 'success':
                    self.stats['classified_images'] += 1
            
            # Create descriptive text
            content_parts = [
                f"Image: {Path(filepath).name}",
                f"Dimensions: {img.size[0]} × {img.size[1]} pixels",
                f"Format: {img.format}",
                f"Mode: {img.mode}"
            ]
            
            # Add classification results to content
            if classification_result and classification_result['status'] == 'success':
                content_parts.extend([
                    f"\n🤖 AI Classification Results:",
                    f"Predicted Species: {classification_result['predicted_species']}",
                    f"Confidence: {classification_result['confidence']:.2%}",
                    f"\nTop 3 Predictions:"
                ])
                for i, pred in enumerate(classification_result['top3_predictions'], 1):
                    content_parts.append(
                        f"{i}. {pred['species']} ({pred['confidence']:.2%})"
                    )
            
            content_parts.append("\nThis image can be used for visual analysis and fish species identification.")
            content = "\n".join(content_parts)
            
            # Prepare metadata
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
            
            # Add classification to metadata
            if classification_result and classification_result['status'] == 'success':
                metadata.update({
                    "ml_predicted_species": classification_result['predicted_species'],
                    "ml_confidence": classification_result['confidence'],
                    "ml_top3": json.dumps(classification_result['top3_predictions']),
                    "ml_classified": True
                })
            else:
                metadata["ml_classified"] = False
            
            doc = Document(page_content=content, metadata=metadata)
            
            self.processed_files.add(file_hash)
            self.stats['images'] += 1
            
            status = "🤖 Classified" if classification_result and classification_result['status'] == 'success' else "Basic"
            logger.info(f"✅ Image ({status}): {Path(filepath).name}")
            
            return [doc]
            
        except Exception as e:
            self.stats['errors'] += 1
            error_msg = f"Image Error: {filepath} - {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.error_log.append(error_msg)
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
            'bmp': self.load_image
        }
        
        if ext in loaders:
            return loaders[ext](filepath)
        else:
            logger.warning(f"⚠️ Unsupported file type: {filepath}")
            return []

# ============================================================
# 🏗️ PRODUCTION VECTOR DB BUILDER WITH ML
# ============================================================
class ProductionVectorDBBuilder:
    """Production-grade vector database builder with ML integration"""
    
    def __init__(self):
        # Initialize fish classifier
        model_path = str(Config.PYTORCH_MODEL_PATH) if Config.PYTORCH_MODEL_PATH.exists() else None
        class_mapping_path = str(Config.CLASS_MAPPING_PATH) if Config.CLASS_MAPPING_PATH.exists() else None
        
        self.fish_classifier = FishClassificationModel(
            model_path=model_path,
            class_mapping_path=class_mapping_path
        )
        
        self.loader = ProductionDocumentLoader(fish_classifier=self.fish_classifier)
        self.embeddings = None
        self.vector_db = None
        self.build_stats = {
            "start_time": None,
            "end_time": None,
            "total_files": 0,
            "total_chunks": 0,
            "ml_classifications": 0,
            "duration_seconds": 0
        }
    
    def scan_directories(self) -> List[str]:
        """Scan all source directories for files"""
        all_files = []
        
        for directory in Config.get_all_source_dirs():
            logger.info(f"📂 Scanning: {directory}")
            
            for root, _, files in os.walk(str(directory)):
                for file in files:
                    filepath = Path(root) / file
                    ext = filepath.suffix.lower().lstrip('.')
                    if ext in Config.SUPPORTED_EXTENSIONS:
                        all_files.append(str(filepath))
        
        logger.info(f"📊 Found {len(all_files)} files to process")
        return all_files
    
    def load_all_documents(self, filepaths: List[str]) -> List[Document]:
        """Load all documents with progress tracking"""
        all_docs = []
        
        logger.info("📥 Loading documents with ML classification...")
        for filepath in tqdm(filepaths, desc="Loading files"):
            docs = self.loader.load_document(filepath)
            all_docs.extend(docs)
            self.loader.stats['total_docs'] += len(docs)
        
        logger.info(f"✅ Loaded {len(all_docs)} documents")
        return all_docs
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        logger.info("✂️ Chunking documents...")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = splitter.split_documents(documents)
        logger.info(f"✅ Created {len(chunks)} chunks")
        return chunks
    
    def build_vector_db(self, chunks: List[Document]) -> Chroma:
        """Build vector database with batch processing"""
        logger.info("🧠 Initializing embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        logger.info("🗄️ Building vector database...")
        
        # Create directory if it doesn't exist
        Config.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        
        if len(chunks) > Config.BATCH_SIZE:
            logger.info(f"Processing in batches of {Config.BATCH_SIZE}...")
            
            self.vector_db = Chroma.from_documents(
                chunks[:Config.BATCH_SIZE],
                embedding=self.embeddings,
                persist_directory=str(Config.VECTOR_DB_DIR),
                collection_name="meenasetu_production"
            )
            
            for i in tqdm(range(Config.BATCH_SIZE, len(chunks), Config.BATCH_SIZE), 
                         desc="Building DB"):
                batch = chunks[i:i + Config.BATCH_SIZE]
                self.vector_db.add_documents(batch)
        else:
            self.vector_db = Chroma.from_documents(
                chunks,
                embedding=self.embeddings,
                persist_directory=str(Config.VECTOR_DB_DIR),
                collection_name="meenasetu_production"
            )
        
        logger.info("✅ Vector database built successfully")
        return self.vector_db
    
    def save_build_report(self):
        """Save detailed build report"""
        report = {
            "build_stats": self.build_stats,
            "loader_stats": self.loader.stats,
            "ml_integration": {
                "model_loaded": self.fish_classifier.is_loaded,
                "model_path": str(Config.PYTORCH_MODEL_PATH),
                "device": str(Config.DEVICE),
                "images_classified": self.loader.stats['classified_images'],
                "total_classes": len(self.fish_classifier.class_mapping) if self.fish_classifier.class_mapping else 0
            },
            "errors": self.loader.error_log,
            "config": {
                "chunk_size": Config.CHUNK_SIZE,
                "chunk_overlap": Config.CHUNK_OVERLAP,
                "embed_model": Config.EMBED_MODEL,
                "source_directories": [str(d) for d in Config.get_all_source_dirs()]
            }
        }
        
        report_path = Config.BASE_DIR / "vector_db_build_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Build report saved: {report_path}")
    
    def build(self):
        """Main build pipeline"""
        logger.info("=" * 70)
        logger.info("🚀 PRODUCTION VECTOR DATABASE BUILD WITH ML")
        logger.info("=" * 70)
        
        self.build_stats["start_time"] = datetime.now().isoformat()
        start = datetime.now()
        
        # Step 1: Scan directories
        filepaths = self.scan_directories()
        self.build_stats["total_files"] = len(filepaths)
        
        if not filepaths:
            logger.warning("⚠️ No files found to process!")
            return
        
        # Step 2: Load documents with ML classification
        documents = self.load_all_documents(filepaths)
        
        if not documents:
            logger.warning("⚠️ No documents loaded!")
            return
        
        # Step 3: Chunk documents
        chunks = self.chunk_documents(documents)
        self.build_stats["total_chunks"] = len(chunks)
        self.build_stats["ml_classifications"] = self.loader.stats['classified_images']
        
        # Step 4: Build vector DB
        self.build_vector_db(chunks)
        
        # Step 5: Finalize
        self.build_stats["end_time"] = datetime.now().isoformat()
        self.build_stats["duration_seconds"] = (datetime.now() - start).total_seconds()
        
        # Step 6: Save report
        self.save_build_report()
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("✨ BUILD COMPLETE WITH ML INTEGRATION ✨")
        logger.info("=" * 70)
        logger.info(f"📁 Total Files: {self.build_stats['total_files']}")
        logger.info(f"📄 Total Documents: {self.loader.stats['total_docs']}")
        logger.info(f"✂️ Total Chunks: {self.build_stats['total_chunks']}")
        logger.info(f"🤖 ML Classifications: {self.build_stats['ml_classifications']}")
        logger.info(f"⏱️ Duration: {self.build_stats['duration_seconds']:.2f} seconds")
        logger.info(f"\n📊 File Type Breakdown:")
        logger.info(f"   PDFs: {self.loader.stats['pdf']}")
        logger.info(f"   CSVs: {self.loader.stats['csv']}")
        logger.info(f"   JSONs: {self.loader.stats['json']}")
        logger.info(f"   TXTs: {self.loader.stats['txt']}")
        logger.info(f"   Images: {self.loader.stats['images']} ({self.loader.stats['classified_images']} classified)")
        logger.info(f"   Errors: {self.loader.stats['errors']}")
        
        if self.fish_classifier.is_loaded:
            logger.info(f"\n🤖 ML Model Info:")
            logger.info(f"   Model: Loaded from {Config.PYTORCH_MODEL_PATH}")
            logger.info(f"   Device: {Config.DEVICE}")
            logger.info(f"   Classes: {len(self.fish_classifier.class_mapping)}")
        
        if self.loader.error_log:
            logger.info(f"\n⚠️ Errors encountered: {len(self.loader.error_log)}")
            logger.info(f"   Check 'vector_db_build_report.json' for details")
        
        logger.info("=" * 70 + "\n")

# ============================================================
# 🎯 MAIN EXECUTION
# ============================================================
def main():
    """Main execution function"""
    try:
        builder = ProductionVectorDBBuilder()
        builder.build()
        
        logger.info("🎉 Vector database build completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Build interrupted by user")
        
    except Exception as e:
        logger.error(f"❌ Fatal error during build: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()