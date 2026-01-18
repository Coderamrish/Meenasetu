import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, Any, Optional
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

# ============================================================
# ⚙️ CONFIGURATION
# ============================================================
class Config:
    """Configuration for ML models"""
    
    # Get the project root directory (adjust as needed)
    BASE_DIR = Path(__file__).parent.parent.parent
    
    # Model paths
    CHECKPOINTS_DIR = BASE_DIR / "training" / "checkpoints"
    
    # Species Classification Models
    SPECIES_MODEL_CONFIGS = {
        'fish_species_model': {
            'path': CHECKPOINTS_DIR / "fish_model.pth",
            'class_mapping': CHECKPOINTS_DIR / "class_mapping1.json"
        }
    }
    
    # Disease Detection Models
    DISEASE_MODEL_CONFIGS = {
        'disease_model_final': {
            'path': CHECKPOINTS_DIR / "final.keras",
            'class_mapping': CHECKPOINTS_DIR / "classes.json"
        }
    }
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 🐟 SPECIES CLASSIFIER
# ============================================================
class FishSpeciesClassifier:
    """Fish species classification model"""
    
    def __init__(self):
        self.device = Config.DEVICE
        self.model = None
        self.class_mapping = None
        self.idx_to_class = None
        
        print("🐟 Initializing Fish Species Classifier...")
        self._load_model()
        self._setup_transforms()
    
    def _load_model(self):
        """Load the species classification model"""
        config = Config.SPECIES_MODEL_CONFIGS['fish_species_model']
        
        if not config['path'].exists():
            raise FileNotFoundError(f"Model file not found: {config['path']}")
        
        # Load checkpoint
        checkpoint = torch.load(str(config['path']), 
                               map_location=self.device, 
                               weights_only=False)
        
        # Extract state dict
        if isinstance(checkpoint, dict):
            state_dict = (checkpoint.get('model_state_dict') or 
                         checkpoint.get('state_dict') or checkpoint)
        else:
            state_dict = checkpoint
        
        # Load class mapping
        self.class_mapping = self._load_class_mapping(checkpoint, str(config['class_mapping']))
        
        if not self.class_mapping:
            raise ValueError("Failed to load class mapping")
        
        # Create and load model
        self.model = self._create_model(len(self.class_mapping))
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).eval()
        
        # Create reverse mapping
        self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
        
        print(f"✅ Loaded species classifier ({len(self.class_mapping)} species)")
    
    def _load_class_mapping(self, checkpoint, external_path: str) -> Optional[Dict]:
        """Load species mapping"""
        class_mapping = None
        
        # Try from checkpoint
        if isinstance(checkpoint, dict):
            if 'species_mapping' in checkpoint:
                sm = checkpoint['species_mapping']
                class_mapping = sm.get('species_to_id') or sm
            elif 'class_mapping' in checkpoint:
                class_mapping = checkpoint['class_mapping']
        
        # Try from external file
        if not class_mapping and Path(external_path).exists():
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
        
        return model
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms"""
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """Predict fish species from image"""
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transforms(img).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Get species name
            species_name = self.idx_to_class.get(predicted.item(), f"Unknown_{predicted.item()}")
            
            # Get top 3 predictions
            top3_prob, top3_indices = torch.topk(probabilities, min(3, probabilities.size(1)))
            top3_predictions = [
                {
                    "species": self.idx_to_class.get(idx.item(), f"Unknown_{idx.item()}"),
                    "confidence": prob.item()
                }
                for prob, idx in zip(top3_prob[0], top3_indices[0])
            ]
            
            return {
                "status": "success",
                "predicted_species": species_name,
                "confidence": float(confidence.item()),
                "top3_predictions": top3_predictions
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def is_loaded(self) -> bool:
        return self.model is not None

# ============================================================
# 🏥 DISEASE DETECTOR
# ============================================================
class FishDiseaseDetector:
    """Fish disease detection model"""
    
    def __init__(self):
        self.model = None
        self.class_mapping = None
        self.id_to_disease = None
        
        print("🏥 Initializing Fish Disease Detector...")
        self._load_model()
    
    def _load_model(self):
        """Load the disease detection model"""
        config = Config.DISEASE_MODEL_CONFIGS['disease_model_final']
        
        if not config['path'].exists():
            raise FileNotFoundError(f"Model file not found: {config['path']}")
        
        # Load class mapping
        self.class_mapping = self._load_class_mapping(str(config['class_mapping']))
        if not self.class_mapping:
            raise ValueError("Failed to load disease class mapping")
        
        # Load Keras model
        self.model = keras.models.load_model(str(config['path']), compile=False)
        
        # Verify dimensions
        model_classes = self.model.output_shape[-1]
        mapping_classes = len(self.class_mapping)
        
        if model_classes != mapping_classes:
            raise ValueError(f"Dimension mismatch: Model={model_classes}, Mapping={mapping_classes}")
        
        # Create reverse mapping
        self.id_to_disease = {v: k for k, v in self.class_mapping.items()}
        
        print(f"✅ Loaded disease detector ({len(self.class_mapping)} diseases)")
    
    def _load_class_mapping(self, mapping_path: str) -> Optional[Dict[str, int]]:
        """Load disease class mapping"""
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mapping_raw = json.load(f)
            
            disease_to_id = {}
            
            # Handle {"0": "disease_name"} format
            for key, value in mapping_raw.items():
                if isinstance(key, str) and key.isdigit():
                    disease_to_id[str(value)] = int(key)
                elif isinstance(value, (int, str)) and str(value).isdigit():
                    disease_to_id[str(key)] = int(value)
            
            return disease_to_id if disease_to_id else None
            
        except Exception as e:
            print(f"Error loading mapping: {e}")
            return None
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """Detect fish disease from image"""
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB').resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = self.model.predict(img_array, verbose=0)
            predicted_id = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_id])
            
            # Get disease name
            disease_name = self.id_to_disease.get(predicted_id, f"Unknown_ID_{predicted_id}")
            
            # Get top 3 predictions
            top3_indices = np.argsort(predictions[0])[-3:][::-1]
            top3_predictions = [
                {
                    "disease": self.id_to_disease.get(int(idx), f"Unknown_{idx}"),
                    "confidence": float(predictions[0][idx])
                }
                for idx in top3_indices
            ]
            
            # Determine if healthy
            is_healthy = any(kw in disease_name.lower() 
                           for kw in ['healthy', 'normal', 'no_disease', 'no disease'])
            
            return {
                "status": "success",
                "predicted_disease": disease_name,
                "confidence": confidence,
                "is_healthy": is_healthy,
                "top3_predictions": top3_predictions
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def is_loaded(self) -> bool:
        return self.model is not None

# ============================================================
# 🎯 MEENASETU AI MODEL MANAGER
# ============================================================
class MeenasetuModels:
    """Manager for all ML models"""
    
    def __init__(self):
        self.species_classifier = None
        self.disease_detector = None
        self._load_models()
    
    def _load_models(self):
        """Load all ML models"""
        try:
            self.species_classifier = FishSpeciesClassifier()
            print("✅ Species classifier loaded")
        except Exception as e:
            print(f"❌ Failed to load species classifier: {e}")
        
        try:
            self.disease_detector = FishDiseaseDetector()
            print("✅ Disease detector loaded")
        except Exception as e:
            print(f"❌ Failed to load disease detector: {e}")
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process image with both models"""
        result = {
            "image": Path(image_path).name,
            "species_detection": None,
            "disease_detection": None,
            "timestamp": str(datetime.now())
        }
        
        # Species classification
        if self.species_classifier and self.species_classifier.is_loaded():
            species_result = self.species_classifier.predict(image_path)
            result["species_detection"] = species_result
        
        # Disease detection
        if self.disease_detector and self.disease_detector.is_loaded():
            disease_result = self.disease_detector.predict(image_path)
            result["disease_detection"] = disease_result
        
        return result
    
    def get_status(self) -> Dict[str, bool]:
        """Get model loading status"""
        return {
            "species_loaded": self.species_classifier is not None and self.species_classifier.is_loaded(),
            "disease_loaded": self.disease_detector is not None and self.disease_detector.is_loaded()
        }

# Global model instance
meenasetu_models = MeenasetuModels()