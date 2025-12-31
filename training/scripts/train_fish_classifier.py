"""
MeenaSetu - Production-Ready Fish Classification Training
Fixed for all PyTorch versions, optimized for small datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import time
import warnings
warnings.filterwarnings('ignore')

class FishDataset(Dataset):
    """Optimized dataset for fish classification"""
    
    def __init__(self, data_df, img_dir, transform=None):
        self.data = data_df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = self.img_dir / row['image_path']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️  Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        return image, row['species_id']


class FishClassifier(nn.Module):
    """Production-ready fish classifier"""
    
    def __init__(self, num_classes, model_type='efficientnet_b0', dropout=0.4):
        super().__init__()
        
        print(f"   📦 Loading {model_type}...")
        
        if model_type == 'efficientnet_b0':
            # Best for small datasets - lightweight and accurate
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
            num_features = self.backbone.classifier[1].in_features
            
            # Custom classifier head
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout * 0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout * 0.3),
                nn.Linear(256, num_classes)
            )
            
        elif model_type == 'mobilenet_v3':
            # Fastest option
            self.backbone = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            )
            num_features = self.backbone.classifier[0].in_features
            
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout * 0.5),
                nn.Linear(256, num_classes)
            )
            
        elif model_type == 'resnet18':
            # Good baseline
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1
            )
            num_features = self.backbone.fc.in_features
            
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout * 0.5),
                nn.Linear(256, num_classes)
            )
        
        else:
            raise ValueError(f"Unknown model: {model_type}")
    
    def forward(self, x):
        return self.backbone(x)


class ProductionTrainer:
    """Production-ready trainer with all fixes"""
    
    def __init__(self, model_type='efficientnet_b0'):
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # GPU setup
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"\n🚀 GPU ENABLED: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f} GB")
            torch.backends.cudnn.benchmark = True
        else:
            print("\n⚠️  CPU MODE - Training will be slow")
            print("   💡 Use Google Colab for free GPU: https://colab.research.google.com/")
        
        # Configuration
        self.config = {
            'data': {
                'fish_csv': 'data/final/merged/fish_mapping_merged_production.csv',
                'images_dir': 'datasets/training/images',
                'img_size': 224,
                'batch_size': 32 if torch.cuda.is_available() else 8,
                'num_workers': 4 if torch.cuda.is_available() else 2
            },
            'model': {
                'type': model_type,
                'dropout': 0.4,
                'freeze_epochs': 3
            },
            'training': {
                'epochs': 50,
                'learning_rate': 0.0001,
                'weight_decay': 0.01,
                'patience': 15,
                'min_images_per_class': 15,
                'use_class_weights': True,
                'label_smoothing': 0.1
            }
        }
        
        self.setup_directories()
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    
    def setup_directories(self):
        for d in ['training/checkpoints', 'training/logs', 'training/outputs']:
            Path(d).mkdir(parents=True, exist_ok=True)
    
    def get_transforms(self, train=True):
        """Augmentation transforms"""
        size = self.config['data']['img_size']
        
        if train:
            return transforms.Compose([
                transforms.Resize((size + 32, size + 32)),
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2)
            ])
        else:
            return transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def prepare_dataset(self):
        """Load dataset from directory"""
        print("\n📁 Scanning for images...")
        img_dir = Path(self.config['data']['images_dir'])
        
        if not img_dir.exists():
            print(f"❌ Directory not found: {img_dir}")
            print("\n📝 Expected structure:")
            print("datasets/training/images/")
            print("├── Species_name_1/")
            print("│   ├── image1.jpg")
            print("│   └── image2.jpg")
            print("└── Species_name_2/")
            return None
        
        data_list = []
        for species_folder in tqdm(list(img_dir.iterdir()), desc="Loading"):
            if not species_folder.is_dir():
                continue
            
            scientific_name = species_folder.name.replace('_', ' ')
            images = (list(species_folder.glob('*.jpg')) + 
                     list(species_folder.glob('*.jpeg')) + 
                     list(species_folder.glob('*.png')))
            
            for img_path in images:
                data_list.append({
                    'image_path': img_path.relative_to(img_dir),
                    'scientific_name': scientific_name
                })
        
        if not data_list:
            print("❌ No images found!")
            return None
        
        df = pd.DataFrame(data_list)
        
        # Filter by minimum images
        min_imgs = self.config['training']['min_images_per_class']
        counts = df['scientific_name'].value_counts()
        valid = counts[counts >= min_imgs].index
        df = df[df['scientific_name'].isin(valid)]
        
        # Create species mapping
        unique_species = sorted(df['scientific_name'].unique())
        self.species_to_id = {s: i for i, s in enumerate(unique_species)}
        self.id_to_species = {i: s for s, i in self.species_to_id.items()}
        df['species_id'] = df['scientific_name'].map(self.species_to_id)
        
        print(f"\n✅ Dataset loaded:")
        print(f"   Total images: {len(df)}")
        print(f"   Species count: {len(unique_species)}")
        print(f"   Avg per species: {len(df)/len(unique_species):.1f}")
        
        # Show species distribution
        print("\n📊 Species distribution:")
        for species in unique_species:
            count = counts[species]
            print(f"   {species}: {count} images")
        
        # Save mapping
        with open('training/outputs/species_mapping.json', 'w') as f:
            json.dump({
                'species_to_id': self.species_to_id,
                'id_to_species': self.id_to_species
            }, f, indent=2)
        
        return df
    
    def create_loaders(self, df):
        """Create train/val dataloaders"""
        train_df, val_df = train_test_split(
            df, test_size=0.2, stratify=df['species_id'], random_state=42
        )
        
        print(f"\n📊 Data split: {len(train_df)} train | {len(val_df)} val")
        
        train_dataset = FishDataset(
            train_df, self.config['data']['images_dir'], 
            self.get_transforms(train=True)
        )
        val_dataset = FishDataset(
            val_df, self.config['data']['images_dir'], 
            self.get_transforms(train=False)
        )
        
        # Weighted sampling for class balance
        sampler = None
        shuffle = True
        
        if self.config['training']['use_class_weights']:
            class_counts = Counter(train_df['species_id'])
            weights = [1.0 / class_counts[sid] for sid in train_df['species_id']]
            sampler = WeightedRandomSampler(weights, len(weights))
            shuffle = False
            print("   Using weighted sampling for class balance")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['data']['batch_size'],
            sampler=sampler,
            shuffle=shuffle,
            num_workers=self.config['data']['num_workers'],
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, val_loader
    
    def initialize_model(self, num_classes):
        """Initialize model and training components"""
        print(f"\n🤖 Initializing model...")
        print(f"   Architecture: {self.model_type}")
        print(f"   Classes: {num_classes}")
        
        self.model = FishClassifier(
            num_classes=num_classes,
            model_type=self.model_type,
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        # Loss with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config['training']['label_smoothing']
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Scheduler - FIXED: No verbose parameter
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
        # Freeze backbone initially
        self.freeze_backbone()
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   ✅ Trainable parameters: {trainable:,}")
    
    def freeze_backbone(self):
        """Freeze backbone for initial training"""
        for name, param in self.model.named_parameters():
            if 'classifier' not in name and 'fc' not in name:
                param.requires_grad = False
        print("   🔒 Backbone frozen (training head only)")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.model.parameters():
            param.requires_grad = True
        print("   🔓 Backbone unfrozen (full training)")
    
    def train_epoch(self, loader, epoch):
        """Train one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/(pbar.n+1):.4f}',
                'acc': f'{100.*correct/total:.1f}%'
            })
        
        return running_loss / len(loader), 100. * correct / total
    
    @torch.no_grad()
    def validate(self, loader):
        """Validate model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return running_loss / len(loader), 100. * correct / total
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print(f"🚀 STARTING TRAINING - {self.model_type.upper()}")
        print("="*70)
        
        df = self.prepare_dataset()
        if df is None:
            return None
        
        train_loader, val_loader = self.create_loaders(df)
        self.initialize_model(len(self.species_to_id))
        
        best_val_acc = 0.0
        patience_counter = 0
        start_time = time.time()
        
        print(f"\n⏱️  Training for {self.config['training']['epochs']} epochs...")
        print(f"   Early stopping patience: {self.config['training']['patience']}")
        print("="*70 + "\n")
        
        for epoch in range(self.config['training']['epochs']):
            # Unfreeze backbone after initial epochs
            if epoch == self.config['model']['freeze_epochs']:
                self.unfreeze_backbone()
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config['training']['learning_rate'] / 10
            
            # Train and validate
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Print results
            print(f"\n📊 Epoch {epoch+1}/{self.config['training']['epochs']} Results:")
            print(f"   Train → Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
            print(f"   Val   → Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
            print(f"   LR: {current_lr:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                improvement = val_acc - best_val_acc
                best_val_acc = val_acc
                patience_counter = 0
                self.save_checkpoint('best_model.pth', epoch, val_acc)
                print(f"   ✅ NEW BEST! Improved by {improvement:.2f}%")
            else:
                patience_counter += 1
                print(f"   📉 No improvement ({patience_counter}/{self.config['training']['patience']})")
            
            # Early stopping
            if patience_counter >= self.config['training']['patience']:
                print(f"\n⏹️  Early stopping triggered!")
                break
            
            print("-" * 70)
        
        # Training complete
        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"⏱️  Total Time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
        print(f"📁 Best model saved: training/checkpoints/best_model.pth")
        
        self.save_checkpoint('final_model.pth', epoch, val_acc)
        self.plot_history()
        
        return best_val_acc
    
    def save_checkpoint(self, filename, epoch, val_acc):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'species_mapping': {
                'species_to_id': self.species_to_id,
                'id_to_species': self.id_to_species
            },
            'config': self.config,
            'model_type': self.model_type
        }
        torch.save(checkpoint, Path('training/checkpoints') / filename)
    
    def plot_history(self):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train', linewidth=2, color='#2E86DE')
        axes[0].plot(self.history['val_loss'], label='Val', linewidth=2, color='#EE5A6F')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(self.history['train_acc'], label='Train', linewidth=2, color='#2E86DE')
        axes[1].plot(self.history['val_acc'], label='Val', linewidth=2, color='#EE5A6F')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # Mark best epoch
        best_epoch = np.argmax(self.history['val_acc'])
        best_acc = self.history['val_acc'][best_epoch]
        axes[1].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5)
        axes[1].text(best_epoch, best_acc, f' Best: {best_acc:.1f}%', 
                    fontsize=10, va='bottom', color='green', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('training/outputs/training_history.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Training curves saved: training/outputs/training_history.png")


if __name__ == "__main__":
    import sys
    
    # Model selection
    available_models = {
        'efficientnet_b0': 'Recommended - Best for small datasets',
        'mobilenet_v3': 'Fastest - Good for deployment',
        'resnet18': 'Baseline - Simple and reliable'
    }
    
    model_type = sys.argv[1] if len(sys.argv) > 1 else 'efficientnet_b0'
    
    if model_type not in available_models:
        print(f"❌ Unknown model: {model_type}")
        print("\n✅ Available models:")
        for name, desc in available_models.items():
            print(f"   • {name}: {desc}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("🐟 MeenaSetu Fish Classification Training")
    print("="*70)
    print(f"🎯 Model: {model_type}")
    print(f"📝 Description: {available_models[model_type]}")
    
    trainer = ProductionTrainer(model_type=model_type)
    trainer.train()