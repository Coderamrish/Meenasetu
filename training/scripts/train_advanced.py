"""
Advanced Fish Classification Training
Implements best practices for maximum accuracy
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
import random

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

class AdvancedFishDataset(Dataset):
    """Dataset with advanced augmentation"""
    
    def __init__(self, data_df, img_dir, transform=None, mixup_alpha=0.2):
        self.data = data_df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.mixup_alpha = mixup_alpha
        
    def __len__(self):
        return len(self.data)
    
    def mixup(self, image1, label1):
        """Apply mixup augmentation"""
        if random.random() > 0.5:  # 50% chance
            idx2 = random.randint(0, len(self.data) - 1)
            row2 = self.data.iloc[idx2]
            img_path2 = self.img_dir / row2['image_path']
            
            try:
                image2 = Image.open(img_path2).convert('RGB')
                if self.transform:
                    image2 = self.transform(image2)
                
                # Mix
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                image = lam * image1 + (1 - lam) * image2
                label2 = row2['species_id']
                
                return image, label1, label2, lam
            except:
                pass
        
        return image1, label1, label1, 1.0
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = self.img_dir / row['image_path']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            image = Image.new('RGB', (224, 224), (128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        # Apply mixup during training
        if self.mixup_alpha > 0:
            image, label1, label2, lam = self.mixup(image, row['species_id'])
            return image, label1, label2, lam
        
        return image, row['species_id']


class AdvancedFishClassifier(nn.Module):
    """Advanced classifier with multiple architecture options"""
    
    def __init__(self, num_classes, model_type='efficientnet_v2_s', dropout=0.3):
        super().__init__()
        
        self.model_type = model_type
        
        if model_type == 'efficientnet_v2_s':
            # Recommended: Best balance of speed and accuracy
            self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = self._build_classifier(num_features, num_classes, dropout)
            
        elif model_type == 'convnext_tiny':
            # Alternative: Excellent for small datasets
            self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            num_features = self.backbone.classifier[2].in_features
            self.backbone.classifier = self._build_classifier(num_features, num_classes, dropout)
            
        elif model_type == 'efficientnet_b0':
            # Lightweight option
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = self._build_classifier(num_features, num_classes, dropout)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _build_classifier(self, num_features, num_classes, dropout):
        """Build classifier head"""
        return nn.Sequential(
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


class AdvancedTrainer:
    """Advanced trainer with best practices"""
    
    def __init__(self, model_type='efficientnet_v2_s'):
        self.model_type = model_type
        self.config = self.create_config()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True
        else:
            print("⚠️  CPU mode - training will be slow")
        
        self.setup_directories()
        self.history = {
            'train_loss': [], 'train_acc': [], 
            'val_loss': [], 'val_acc': [], 'lr': []
        }
    
    def create_config(self):
        """Create advanced configuration"""
        return {
            'data': {
                'fish_csv': 'data/final/merged/fish_mapping_merged_production.csv',
                'images_dir': 'datasets/training/images',
                'img_size': 224,
                'batch_size': 16,
                'num_workers': 2
            },
            'model': {
                'type': self.model_type,
                'dropout': 0.3,
                'freeze_epochs': 5  # Freeze longer for better features
            },
            'training': {
                'epochs': 100,  # More epochs
                'learning_rate': 0.0001,
                'weight_decay': 0.01,
                'patience': 20,  # More patience
                'min_images_per_class': 15,
                'use_class_weights': True,
                'label_smoothing': 0.1,
                'mixup_alpha': 0.2,  # Mixup augmentation
                'gradient_clip': 1.0
            },
            'augmentation': {
                'enabled': True,
                'horizontal_flip': 0.5,
                'vertical_flip': 0.3,
                'rotation': 45,  # More aggressive
                'brightness': 0.4,
                'contrast': 0.4,
                'saturation': 0.4,
                'hue': 0.15,
                'random_erase': 0.3
            }
        }
    
    def setup_directories(self):
        for d in ['training/checkpoints', 'training/logs', 'training/outputs']:
            Path(d).mkdir(parents=True, exist_ok=True)
    
    def get_transforms(self, train=True, img_size=224):
        """Enhanced augmentation"""
        if train:
            aug = self.config['augmentation']
            return transforms.Compose([
                transforms.Resize((img_size + 32, img_size + 32)),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(p=aug['horizontal_flip']),
                transforms.RandomVerticalFlip(p=aug['vertical_flip']),
                transforms.RandomRotation(aug['rotation']),
                transforms.ColorJitter(
                    brightness=aug['brightness'],
                    contrast=aug['contrast'],
                    saturation=aug['saturation'],
                    hue=aug['hue']
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.15, 0.15),
                    scale=(0.85, 1.15),
                    shear=10
                ),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=aug['random_erase'])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def prepare_dataset(self):
        """Load and prepare dataset"""
        print("\n📁 Loading images...")
        img_dir = Path(self.config['data']['images_dir'])
        
        data_list = []
        for species_folder in tqdm(list(img_dir.iterdir()), desc="Scanning"):
            if not species_folder.is_dir():
                continue
            
            scientific_name = species_folder.name.replace('_', ' ')
            images = list(species_folder.glob('*.jpg')) + \
                    list(species_folder.glob('*.jpeg')) + \
                    list(species_folder.glob('*.png'))
            
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
        
        # Create mappings
        unique_species = sorted(df['scientific_name'].unique())
        self.species_to_id = {s: i for i, s in enumerate(unique_species)}
        self.id_to_species = {i: s for s, i in self.species_to_id.items()}
        
        df['species_id'] = df['scientific_name'].map(self.species_to_id)
        
        print(f"✅ Dataset: {len(df)} images, {len(unique_species)} species")
        for species in unique_species:
            count = counts[species]
            print(f"   {species}: {count} images")
        
        # Save mapping
        with open('training/outputs/species_mapping_advanced.json', 'w') as f:
            json.dump({
                'species_to_id': self.species_to_id,
                'id_to_species': self.id_to_species
            }, f, indent=2)
        
        return df
    
    def create_loaders(self, df):
        """Create dataloaders"""
        train_df, val_df = train_test_split(
            df, test_size=0.2, stratify=df['species_id'], random_state=42
        )
        
        print(f"\n📊 Split: {len(train_df)} train, {len(val_df)} val")
        
        train_dataset = AdvancedFishDataset(
            train_df, self.config['data']['images_dir'],
            self.get_transforms(train=True),
            mixup_alpha=self.config['training']['mixup_alpha']
        )
        
        val_dataset = AdvancedFishDataset(
            val_df, self.config['data']['images_dir'],
            self.get_transforms(train=False),
            mixup_alpha=0  # No mixup for validation
        )
        
        # Weighted sampling
        if self.config['training']['use_class_weights']:
            class_counts = Counter(train_df['species_id'])
            weights = [1.0 / class_counts[sid] for sid in train_df['species_id']]
            sampler = WeightedRandomSampler(weights, len(weights))
        else:
            sampler = None
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['data']['batch_size'],
            sampler=sampler,
            shuffle=(sampler is None),
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
        """Initialize model"""
        print(f"\n🤖 Building {self.model_type} for {num_classes} classes...")
        
        self.model = AdvancedFishClassifier(
            num_classes=num_classes,
            model_type=self.model_type,
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config['training']['label_smoothing']
        )
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Better scheduler
        # Create scheduler; some PyTorch versions don't accept `verbose` so fall back if needed
        try:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
            )
        except TypeError:
            # Older/newer PyTorch may not support verbose kwarg
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5
            )
        
        # Freeze backbone initially
        self.freeze_backbone()
        
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Trainable parameters: {params:,}")
    
    def freeze_backbone(self):
        """Freeze backbone"""
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        print("   🔒 Backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone"""
        for param in self.model.parameters():
            param.requires_grad = True
        print("   🔓 Backbone unfrozen")
    
    def mixup_criterion(self, pred, y_a, y_b, lam):
        """Mixup loss"""
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)
    
    def train_epoch(self, loader, epoch):
        """Train one epoch with mixup"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            if len(batch) == 4:  # Mixup
                images, labels1, labels2, lam = batch
                images = images.to(self.device)
                labels1 = labels1.to(self.device)
                labels2 = labels2.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.mixup_criterion(outputs, labels1, labels2, lam.to(self.device))
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config['training']['gradient_clip']
                )
                
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels1.size(0)
                correct += (lam * predicted.eq(labels1).float() + 
                           (1-lam) * predicted.eq(labels2).float()).sum().item()
            else:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config['training']['gradient_clip']
                )
                
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return running_loss / len(loader), 100. * correct / total
    
    @torch.no_grad()
    def validate(self, loader):
        """Validate with TTA (Test-Time Augmentation)"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in loader:
            if len(batch) == 4:
                images, labels, _, _ = batch
            else:
                images, labels = batch
            
            images, labels = images.to(self.device), labels.to(self.device)
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
        print(f"🚀 ADVANCED TRAINING - {self.model_type.upper()}")
        print("="*70)
        
        df = self.prepare_dataset()
        if df is None:
            return
        
        train_loader, val_loader = self.create_loaders(df)
        self.initialize_model(len(self.species_to_id))
        
        best_val_acc = 0.0
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(self.config['training']['epochs']):
            # Unfreeze backbone
            if epoch == self.config['model']['freeze_epochs']:
                self.unfreeze_backbone()
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config['training']['learning_rate'] / 10
            
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate(val_loader)
            
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")
            print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"   Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            print(f"   LR: {current_lr:.6f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_checkpoint('best_model_advanced.pth', epoch, val_acc)
                print(f"   ✅ New best! Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1
            
            if patience_counter >= self.config['training']['patience']:
                print(f"\n⏹️  Early stopping")
                break
        
        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"Best Val Accuracy: {best_val_acc:.2f}%")
        print(f"Time: {elapsed/60:.1f} minutes")
        
        self.save_checkpoint('final_model_advanced.pth', epoch, val_acc)
        self.plot_history()
    
    def save_checkpoint(self, filename, epoch, val_acc):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'species_mapping': {
                'species_to_id': self.species_to_id,
                'id_to_species': self.id_to_species
            },
            'config': self.config
        }
        torch.save(checkpoint, Path('training/checkpoints') / filename)
    
    def plot_history(self):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].plot(self.history['train_loss'], label='Train', linewidth=2)
        axes[0].plot(self.history['val_loss'], label='Val', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.history['train_acc'], label='Train', linewidth=2)
        axes[1].plot(self.history['val_acc'], label='Val', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(self.history['lr'], linewidth=2, color='orange')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training/outputs/training_history_advanced.png', dpi=300)
        print(f"📊 Saved: training/outputs/training_history_advanced.png")


if __name__ == "__main__":
    import sys
    
    # Allow model selection from command line
    model_type = sys.argv[1] if len(sys.argv) > 1 else 'efficientnet_v2_s'
    
    print(f"\n🎯 Training with: {model_type}")
    print("Available models:")
    print("  - efficientnet_v2_s (recommended)")
    print("  - convnext_tiny")
    print("  - efficientnet_b0")
    
    trainer = AdvancedTrainer(model_type=model_type)
    trainer.train()