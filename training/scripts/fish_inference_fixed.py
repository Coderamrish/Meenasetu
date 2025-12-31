"""
MeenaSetu - Fish Classification Inference System (Fixed)
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

class ImprovedFishClassifier(nn.Module):
    """Improved classifier - MUST match training architecture exactly"""
    
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        
        # Use EfficientNet-B0 (same as training)
        self.backbone = models.efficientnet_b0(weights=None)
        
        # Replace classifier with improved head (EXACT match to training)
        num_features = 1280  # EfficientNet-B0 feature size
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout/2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout/3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


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
        
        # Get config to determine model architecture
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})
        model_type = model_config.get('type', 'efficientnet_b0')
        dropout = model_config.get('dropout', 0.5)
        
        print(f"   Model type: {model_type}")
        print(f"   Classes: {num_classes}")
        
        # Build model with EXACT architecture from training
        self.model = ImprovedFishClassifier(
            num_classes=num_classes,
            dropout=dropout
        ).to(self.device)
        
        # Load weights
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print("   ✅ Model loaded successfully (strict mode)")
        except Exception as e:
            print(f"   ⚠️  Strict load failed: {e}")
            # Try non-strict as fallback
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("   ⚠️  Loaded with strict=False")
        
        self.model.eval()
        
        val_acc = checkpoint.get('val_acc', 0)
        print(f"   ✅ Validation Accuracy: {val_acc:.2f}%")
    
    def load_fish_database(self):
        """Load fish metadata database"""
        fish_db_path = 'data/final/merged/fish_mapping_merged_production.csv'
        
        if not Path(fish_db_path).exists():
            print(f"   ⚠️  Fish database not found: {fish_db_path}")
            self.fish_db = pd.DataFrame()
            return
        
        self.fish_db = pd.read_csv(fish_db_path)
        self.fish_db['scientific_name_key'] = self.fish_db['scientific_name'].str.lower()
        print(f"   📚 Loaded {len(self.fish_db)} fish records")
    
    def setup_transforms(self):
        """Setup image preprocessing - MUST match training"""
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
            # Get species name from mapping
            id_to_species = self.species_mapping.get('id_to_species', {})
            
            # Handle both int and string keys
            species_name = id_to_species.get(str(idx), id_to_species.get(int(idx), f"Unknown_{idx}"))
            
            # Get metadata from database
            if len(self.fish_db) > 0:
                fish_data = self.fish_db[
                    self.fish_db['scientific_name_key'] == species_name.lower()
                ]
                
                if len(fish_data) > 0:
                    fish_info = fish_data.iloc[0].to_dict()
                else:
                    fish_info = {'scientific_name': species_name}
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
        print("   python training/scripts/train_fish_classifier_improved.py")
        return
    
    # Initialize inference
    try:
        classifier = FishClassifierInference(str(model_path))
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        classifier.visualize_prediction(img_path, output_path)
        
        print()


if __name__ == "__main__":
    demo_inference()