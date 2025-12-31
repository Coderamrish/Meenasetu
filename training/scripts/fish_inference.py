"""
MeenaSetu - Fish Classification Inference System
Load trained model and identify fish from images
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
from pathlib import Path
import pandas as pd
import numpy as np

class FishClassifierInference:
    """Inference system for fish classification"""
    
    def __init__(self, checkpoint_path='training/checkpoints/best_model.pth'):
        """
        Initialize inference system
        
        Args:
            checkpoint_path: Path to trained model checkpoint
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        
        # Load checkpoint
        self.load_checkpoint()
        
        # Load fish database
        self.load_fish_database()
        
        # Setup transforms
        self.setup_transforms()
        
        print(f"✅ Inference system ready!")
        print(f"   Device: {self.device}")
        print(f"   Species: {len(self.species_mapping['id_to_species'])}")
    
    def load_checkpoint(self):
        """Load trained model"""
        print(f"📦 Loading checkpoint: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Get species mapping
        self.species_mapping = checkpoint['species_mapping']
        num_classes = len(self.species_mapping['species_to_id'])

        # Determine model type from saved config (fallback to resnet50)
        model_type = 'resnet50'
        try:
            model_type = checkpoint.get('config', {}).get('model', {}).get('type', model_type)
        except Exception:
            pass

        # Rebuild model to match training architecture
        self.model = self.build_model(num_classes, model_type)

        # Try strict load first; if it fails, attempt non-strict load and warn
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"⚠️  Warning: strict state_dict load failed: {e}")
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("   ⚠️  Loaded model with strict=False (some keys ignored)")
            except Exception as e2:
                raise RuntimeError(f"Failed to load model weights: {e2}")

        self.model.eval()
        
        print(f"   ✅ Model loaded (Val Acc: {checkpoint.get('val_acc', 0):.2f}%) using {model_type}")
    
    def build_model(self, num_classes, model_type='resnet50'):
        """Build model architecture matching training options
        Supported: 'efficientnet_v2_s', 'mobilenet_v3', 'resnet50', 'convnext_tiny'
        """
        model_type = model_type or 'resnet50'

        if model_type == 'efficientnet_v2_s':
            # Matches training backbone
            model = models.efficientnet_v2_s(weights=None)
            # classifier structure differs by torchvision version; access safely
            try:
                num_features = model.classifier[1].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(num_features, num_classes)
                )
            except Exception:
                # Fallback to a simple head
                num_features = model.classifier.in_features if hasattr(model.classifier, 'in_features') else 1280
                model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_features, num_classes))

        elif model_type == 'mobilenet_v3':
            model = models.mobilenet_v3_small(weights=None)
            try:
                num_features = model.classifier[3].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(num_features, num_classes)
                )
            except Exception:
                num_features = model.classifier.in_features if hasattr(model.classifier, 'in_features') else 576
                model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_features, num_classes))

        elif model_type == 'convnext_tiny':
            model = models.convnext_tiny(weights=None)
            try:
                num_features = model.classifier[2].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(num_features, num_classes)
                )
            except Exception:
                num_features = model.classifier.in_features if hasattr(model.classifier, 'in_features') else 768
                model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_features, num_classes))

        else:
            # Default: resnet50 (baseline)
            model = models.resnet50(weights=None)
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )

        return model.to(self.device)
    
    def load_fish_database(self):
        """Load fish metadata database"""
        fish_db_path = 'data/final/merged/fish_mapping_merged_production.csv'
        self.fish_db = pd.read_csv(fish_db_path)
        self.fish_db['scientific_name_key'] = self.fish_db['scientific_name'].str.lower()
    
    def setup_transforms(self):
        """Setup image preprocessing"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path):
        """Preprocess image for inference"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def predict(self, image_path, top_k=5):
        """
        Predict fish species from image
        
        Args:
            image_path: Path to image file
            top_k: Return top K predictions
            
        Returns:
            List of predictions with probabilities and metadata
        """
        # Preprocess
        image_tensor = self.preprocess_image(image_path)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top K predictions
        probs, indices = torch.topk(probabilities, top_k)
        probs = probs.cpu().numpy()[0]
        indices = indices.cpu().numpy()[0]
        
        # Build results
        results = []
        for prob, idx in zip(probs, indices):
            # Handle id_to_species keys saved as ints (checkpoint) or strings (json)
            id_to_species = self.species_mapping.get('id_to_species', {})
            # Try int key first, then string fallback
            try:
                species_name = id_to_species.get(int(idx), id_to_species.get(str(idx), None))
            except Exception:
                species_name = id_to_species.get(str(idx), None)
            if species_name is None:
                # Fallback to using index as species identifier
                species_name = str(idx)
            
            # Get metadata from database
            fish_data = self.fish_db[
                self.fish_db['scientific_name_key'] == species_name.lower()
            ]
            
            if len(fish_data) > 0:
                fish_info = fish_data.iloc[0].to_dict()
            else:
                fish_info = {'scientific_name': species_name}
            
            results.append({
                'scientific_name': species_name,
                'confidence': float(prob),
                'confidence_pct': f"{float(prob)*100:.1f}%",
                'local_name': fish_info.get('local_name', 'N/A'),
                'common_name': fish_info.get('common_name', 'N/A'),
                'family': fish_info.get('family', 'N/A'),
                'max_size': fish_info.get('max_size', 'N/A'),
                'iucn_status': fish_info.get('iucn_status', 'N/A')
            })
        
        return results
    
    def predict_batch(self, image_paths, batch_size=32):
        """Predict multiple images"""
        results = []
        
        for img_path in image_paths:
            pred = self.predict(img_path, top_k=1)
            results.append({
                'image': str(img_path),
                'prediction': pred[0]
            })
        
        return results
    
    def visualize_prediction(self, image_path, save_path=None):
        """Visualize prediction with image"""
        import matplotlib.pyplot as plt
        
        # Get prediction
        predictions = self.predict(image_path, top_k=3)
        
        # Load image
        image = Image.open(image_path)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Show image
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title('Input Image', fontsize=14, fontweight='bold')
        
        # Show predictions
        ax2.axis('off')
        
        y_pos = 0.9
        for i, pred in enumerate(predictions):
            confidence = pred['confidence']
            color = 'green' if i == 0 else 'gray'
            
            # Title
            ax2.text(0.05, y_pos, f"#{i+1}: {pred['scientific_name']}", 
                    fontsize=12, fontweight='bold', color=color)
            y_pos -= 0.08
            
            # Details
            details = [
                f"Confidence: {pred['confidence_pct']}",
                f"Local Name: {pred['local_name']}",
                f"Family: {pred['family']}",
                f"Status: {pred['iucn_status']}"
            ]
            
            for detail in details:
                ax2.text(0.1, y_pos, detail, fontsize=10, color=color)
                y_pos -= 0.06
            
            y_pos -= 0.05
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_title('Top 3 Predictions', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved visualization: {save_path}")
        else:
            plt.show()
        
        return fig


def demo_inference():
    """Demo inference on sample images"""
    print("\n" + "="*70)
    print("🐟 Fish Classification Inference Demo")
    print("="*70)
    
    # Check if model exists
    model_path = Path('training/checkpoints/best_model.pth')
    
    if not model_path.exists():
        print("\n❌ Model not found!")
        print(f"Expected location: {model_path}")
        print("\n💡 Please train the model first:")
        print("   python training/scripts/train_fish_classifier.py")
        return
    
    # Initialize inference
    try:
        classifier = FishClassifierInference(str(model_path))
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        return
    
    # Find sample images
    img_dir = Path('datasets/training/images')
    sample_images = []
    
    if img_dir.exists():
        # Get first image from each species folder
        for species_folder in img_dir.iterdir():
            if species_folder.is_dir():
                images = list(species_folder.glob('*.jpg')) + \
                        list(species_folder.glob('*.jpeg')) + \
                        list(species_folder.glob('*.png'))
                if images:
                    sample_images.append(images[0])
                if len(sample_images) >= 3:
                    break
    
    if not sample_images:
        print("\n⚠️  No sample images found")
        print(f"Place images in: {img_dir}")
        return
    
    # Run inference on samples
    print(f"\n🔍 Running inference on {len(sample_images)} images...\n")
    
    for img_path in sample_images:
        print(f"📸 Image: {img_path.name}")
        print("-" * 70)
        
        # Predict
        predictions = classifier.predict(img_path, top_k=3)
        
        for i, pred in enumerate(predictions, 1):
            print(f"\n   {i}. {pred['scientific_name']} ({pred['confidence_pct']})")
            print(f"      Local: {pred['local_name']}")
            print(f"      Family: {pred['family']}")
            print(f"      Status: {pred['iucn_status']}")
        
        # Save visualization
        output_path = Path('training/outputs') / f"prediction_{img_path.stem}.png"
        classifier.visualize_prediction(img_path, output_path)
        
        print()


def create_web_api_for_inference():
    """Create Flask API for inference"""
    api_code = '''
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from pathlib import Path
from fish_inference import FishClassifierInference

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Load model
classifier = FishClassifierInference()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/identify', methods=['POST'])
def identify_fish():
    """Identify fish from uploaded image"""
    
    # Check if file is present
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Get top K predictions
        top_k = request.form.get('top_k', 5, type=int)
        predictions = classifier.predict(filepath, top_k=top_k)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/api/species', methods=['GET'])
def get_species_list():
    """Get list of identifiable species"""
    species_list = [
        {'id': k, 'name': v}
        for k, v in classifier.species_mapping['id_to_species'].items()
    ]
    return jsonify({
        'count': len(species_list),
        'species': species_list
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier.model is not None,
        'species_count': len(classifier.species_mapping['id_to_species'])
    })

if __name__ == '__main__':
    print("\\n🐟 Fish Identification API")
    print("🌐 http://localhost:5001")
    print("\\n📡 Endpoints:")
    print("   POST /api/identify - Upload image for identification")
    print("   GET  /api/species  - Get list of species")
    print("   GET  /health       - Health check")
    app.run(debug=True, port=5001)
'''
    
    with open('training/scripts/inference_api.py', 'w') as f:
        f.write(api_code)
    
    print("✅ Created inference API: training/scripts/inference_api.py")
    print("\n🚀 Run with: python training/scripts/inference_api.py")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--create-api':
        create_web_api_for_inference()
    else:
        demo_inference()