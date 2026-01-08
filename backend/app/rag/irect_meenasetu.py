import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
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

# ============================================================
# ⚙️ CONFIGURATION
# ============================================================
class DirectConfig:
    """Direct mode configuration"""
    
    BASE_DIR = Path(r"C:\Users\AMRISH\Documents\Meenasetu")
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
            'class_mapping': CHECKPOINTS_DIR / "classes2.json"
        }
    }
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 🐟 DIRECT SPECIES CLASSIFIER
# ============================================================
class DirectFishClassifier:
    """Direct species classification without RAG"""
    
    def __init__(self):
        self.device = DirectConfig.DEVICE
        self.models = {}
        self.class_mappings = {}
        
        print("🐟 Initializing Direct Species Classifier...")
        self._load_models()
        self._setup_transforms()
    
    def _load_models(self):
        """Load species models"""
        for model_name, config in DirectConfig.SPECIES_MODEL_CONFIGS.items():
            if not config['path'].exists():
                print(f"  ⚠️ Skipping {model_name} (file not found)")
                continue
                
            try:
                checkpoint = torch.load(str(config['path']), 
                                       map_location=self.device, 
                                       weights_only=False)
                
                state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint
                
                # Load class mapping
                class_mapping = self._load_class_mapping(checkpoint, str(config['class_mapping']))
                if not class_mapping:
                    continue
                
                # Create and load model
                model = self._create_model(len(class_mapping))
                model.load_state_dict(state_dict, strict=False)
                model.to(self.device).eval()
                
                self.models[model_name] = model
                self.class_mappings[model_name] = class_mapping
                
                print(f"  ✅ Loaded {model_name} ({len(class_mapping)} species)")
                
            except Exception as e:
                print(f"  ❌ Failed {model_name}: {e}")
    
    def _load_class_mapping(self, checkpoint, external_path: str) -> Optional[Dict]:
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
        
        return model
    
    def _setup_transforms(self):
        """Setup preprocessing"""
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def classify_image(self, image_path: str) -> Dict[str, Any]:
        """Classify fish species from image"""
        if not self.models:
            return {"status": "error", "message": "No models loaded"}
        
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transforms(img).unsqueeze(0).to(self.device)
            
            # Use first available model
            model_name = list(self.models.keys())[0]
            model = self.models[model_name]
            class_mapping = self.class_mappings[model_name]
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            idx_to_class = {v: k for k, v in class_mapping.items()}
            species_name = idx_to_class.get(predicted.item(), f"Unknown_{predicted.item()}")
            
            # Get top 3 predictions
            top3_prob, top3_indices = torch.topk(probabilities, min(3, probabilities.size(1)))
            top3_predictions = [{
                "species": idx_to_class.get(idx.item(), f"Unknown_{idx.item()}"),
                "confidence": prob.item()
            } for prob, idx in zip(top3_prob[0], top3_indices[0])]
            
            result = {
                "status": "success",
                "predicted_species": species_name,
                "confidence": confidence.item(),
                "top3_predictions": top3_predictions,
                "model_used": model_name
            }
            
            print(f"🐟 Species: {species_name} ({confidence.item():.1%})")
            return result
            
        except Exception as e:
            print(f"❌ Classification error: {e}")
            return {"status": "error", "message": str(e)}
    
    @property
    def is_loaded(self) -> bool:
        return len(self.models) > 0

# ============================================================
# 🏥 DIRECT DISEASE DETECTOR
# ============================================================
class DirectDiseaseDetector:
    """Direct disease detection without RAG"""
    
    def __init__(self):
        self.models = {}
        self.class_mappings = {}
        
        if not TENSORFLOW_AVAILABLE:
            print("⚠️ TensorFlow not available - disease detection disabled")
            return
        
        print("🏥 Initializing Direct Disease Detector...")
        self._load_models()
    
    def _load_models(self):
        """Load disease models - SIMPLIFIED"""
        for model_name, config in DirectConfig.DISEASE_MODEL_CONFIGS.items():
            model_path = config['path']
            mapping_path = config['class_mapping']
            
            if not model_path.exists():
                print(f"  ⚠️ Model file not found: {model_path}")
                continue
            
            try:
                # Load class mapping (SIMPLE VERSION)
                print(f"  📋 Loading mapping from: {mapping_path}")
                with open(str(mapping_path), 'r', encoding='utf-8') as f:
                    mapping_raw = json.load(f)
                
                # Simple mapping parser
                class_mapping = {}
                for key, value in mapping_raw.items():
                    if isinstance(key, str) and key.isdigit():
                        class_mapping[str(value)] = int(key)
                    elif isinstance(value, (int, str)) and str(value).isdigit():
                        class_mapping[str(key)] = int(value)
                
                if not class_mapping:
                    print(f"  ❌ Failed to parse mapping")
                    continue
                
                # Load Keras model
                print(f"  🤖 Loading model: {model_name}")
                model = keras.models.load_model(str(model_path), compile=False)
                
                # Verify dimensions
                model_classes = model.output_shape[-1]
                mapping_classes = len(class_mapping)
                
                if model_classes != mapping_classes:
                    print(f"  ❌ Dimension mismatch: Model={model_classes}, Mapping={mapping_classes}")
                    continue
                
                self.models[model_name] = model
                self.class_mappings[model_name] = class_mapping
                print(f"  ✅ Loaded {model_name} ({len(class_mapping)} diseases)")
                
            except Exception as e:
                print(f"  ❌ Failed {model_name}: {e}")
    
    def detect_disease(self, image_path: str) -> Dict[str, Any]:
        """Detect fish disease from image"""
        if not self.models:
            return {
                "status": "unavailable", 
                "message": "Disease detection not available"
            }
        
        try:
            # Preprocess image
            img = Image.open(image_path).convert('RGB').resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Use first available model
            model_name = list(self.models.keys())[0]
            model = self.models[model_name]
            class_mapping = self.class_mappings[model_name]
            
            # Reverse mapping for prediction
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
            
            result = {
                "status": "success",
                "predicted_disease": disease_name,
                "confidence": confidence,
                "is_healthy": is_healthy,
                "top3_predictions": top3_predictions,
                "model_used": model_name,
                "total_classes": len(class_mapping)
            }
            
            health_status = "🟢 Healthy" if is_healthy else "🔴 Disease"
            print(f"🏥 Disease: {disease_name} ({confidence:.1%}) - {health_status}")
            return result
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return {"status": "error", "message": str(e)}
    
    @property
    def is_loaded(self) -> bool:
        return len(self.models) > 0

# ============================================================
# 🎯 DIRECT MEENASETU AI (NO RAG)
# ============================================================
class DirectMeenasetuAI:
    """Direct MeenaSetu AI with ML models only (no RAG)"""
    
    def __init__(self):
        print("=" * 60)
        print("🎯 DIRECT MEENASETU AI - ML MODELS ONLY")
        print("=" * 60)
        
        # Initialize ML models
        self.species_classifier = DirectFishClassifier()
        self.disease_detector = DirectDiseaseDetector()
        
        # Statistics
        self.stats = {
            "images_processed": 0,
            "species_detections": 0,
            "disease_detections": 0
        }
        
        self._print_status()
    
    def _print_status(self):
        """Print system status"""
        print(f"\n📊 SYSTEM STATUS:")
        print(f"  🐟 Species Classifier: {'✅ Ready' if self.species_classifier.is_loaded else '❌ Not loaded'}")
        print(f"  🏥 Disease Detector: {'✅ Ready' if self.disease_detector.is_loaded else '❌ Not loaded'}")
        
        if self.species_classifier.is_loaded:
            model_name = list(self.species_classifier.models.keys())[0]
            species_count = len(self.species_classifier.class_mappings[model_name])
            print(f"     - Model: {model_name}")
            print(f"     - Species: {species_count}")
        
        if self.disease_detector.is_loaded:
            model_name = list(self.disease_detector.models.keys())[0]
            disease_count = len(self.disease_detector.class_mappings[model_name])
            print(f"     - Model: {model_name}")
            print(f"     - Diseases: {disease_count}")
            print(f"     - Classes: {list(self.disease_detector.class_mappings[model_name].keys())}")
    
    def process_image(self, image_path: str, check_disease: bool = True) -> Dict[str, Any]:
        """Process image for species and disease detection"""
        print(f"\n📸 Processing image: {Path(image_path).name}")
        
        if not Path(image_path).exists():
            return {"status": "error", "message": "Image file not found"}
        
        result = {
            "image": Path(image_path).name,
            "timestamp": str(datetime.now()),
            "species_detection": None,
            "disease_detection": None
        }
        
        # Species classification
        if self.species_classifier.is_loaded:
            species_result = self.species_classifier.classify_image(image_path)
            result["species_detection"] = species_result
            if species_result["status"] == "success":
                self.stats["species_detections"] += 1
        
        # Disease detection (optional)
        if check_disease and self.disease_detector.is_loaded:
            disease_result = self.disease_detector.detect_disease(image_path)
            result["disease_detection"] = disease_result
            if disease_result.get("status") == "success":
                self.stats["disease_detections"] += 1
        
        self.stats["images_processed"] += 1
        
        print(f"✅ Image processed successfully")
        return result
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        return {
            **self.stats,
            "species_loaded": self.species_classifier.is_loaded,
            "disease_loaded": self.disease_detector.is_loaded
        }

# ============================================================
# 🧪 TEST DIRECT SYSTEM
# ============================================================
def test_direct_system():
    """Test the direct MeenaSetu AI system"""
    print("\n" + "=" * 60)
    print("🧪 TESTING DIRECT MEENASETU AI")
    print("=" * 60)
    
    try:
        # Initialize
        ai = DirectMeenasetuAI()
        
        # Test with a sample image (update path as needed)
        sample_images = [
            r"C:\Users\AMRISH\Documents\Meenasetu\test_fish.jpg",  # Update this path
            r"C:\Users\AMRISH\Documents\Meenasetu\test_disease.jpg"  # Update this path
        ]
        
        for img_path in sample_images:
            if Path(img_path).exists():
                print(f"\n🔍 Testing with: {Path(img_path).name}")
                result = ai.process_image(img_path, check_disease=True)
                
                # Print results
                if result["species_detection"] and result["species_detection"]["status"] == "success":
                    sp = result["species_detection"]
                    print(f"🐟 Species: {sp['predicted_species']} ({sp['confidence']:.1%})")
                
                if result["disease_detection"] and result["disease_detection"]["status"] == "success":
                    dd = result["disease_detection"]
                    status = "Healthy 🟢" if dd.get('is_healthy') else f"Disease 🔴: {dd['predicted_disease']}"
                    print(f"🏥 Health: {status} ({dd['confidence']:.1%})")
            else:
                print(f"⚠️ Image not found: {img_path}")
        
        # Print statistics
        print(f"\n📊 Final Statistics:")
        stats = ai.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n✅ Direct system test complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================
# 🚀 SIMPLE COMMAND LINE INTERFACE
# ============================================================
def simple_cli():
    """Simple command-line interface"""
    print("🚀 MEENASETU AI - DIRECT MODE")
    print("Commands: detect <image_path> | stats | exit")
    
    ai = DirectMeenasetuAI()
    
    while True:
        try:
            command = input("\n>>> ").strip().lower()
            
            if command.startswith("detect "):
                image_path = command[7:].strip()
                result = ai.process_image(image_path)
                
                if result["species_detection"]:
                    sp = result["species_detection"]
                    if sp["status"] == "success":
                        print(f"🐟 Species: {sp['predicted_species']} ({sp['confidence']:.1%})")
                
                if result["disease_detection"]:
                    dd = result["disease_detection"]
                    if dd["status"] == "success":
                        health = "Healthy 🟢" if dd.get('is_healthy') else f"Disease 🔴"
                        print(f"🏥 {health}: {dd['predicted_disease']} ({dd['confidence']:.1%})")
            
            elif command == "stats":
                stats = ai.get_statistics()
                for key, value in stats.items():
                    print(f"{key}: {value}")
            
            elif command in ["exit", "quit"]:
                print("👋 Goodbye!")
                break
            
            else:
                print("Commands: detect <image_path> | stats | exit")
                
        except Exception as e:
            print(f"Error: {e}")

# ============================================================
# 🎯 MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    # Option 1: Run test
    test_direct_system()
    
    # Option 2: Run CLI (uncomment to use)
    # simple_cli()