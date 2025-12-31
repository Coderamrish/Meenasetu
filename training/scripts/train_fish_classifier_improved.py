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

class ImprovedFishDataset(Dataset):
    """Enhanced dataset with better augmentation"""
    
    def __init__(self, data_df, img_dir, transform=None, use_strong_aug=False):
        self.data = data_df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.use_strong_aug = use_strong_aug
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = self.img_dir / row['image_path']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), (128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        return image, row['species_id']

class ImprovedFishClassifier(nn.Module):
    """Improved classifier with better architecture for small datasets"""
    
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        
        # Use EfficientNet-B0 (smaller, better for limited data)
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Replace classifier with improved head
        num_features = self.backbone.classifier[1].in_features
        
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

class ImprovedTrainer:
    """Enhanced trainer with fixes for small datasets"""
    
    def __init__(self, config_path='training/configs/fish_config_improved.json'):
        self.config = self.load_improved_config(config_path)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True
        else:
            print("⚠️  CPU mode - training will be slow")
        
        self.setup_directories()
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    
    def load_improved_config(self, config_path):
        """Load config optimized for small datasets"""
        config = {
            'data': {
                'fish_csv': 'data/final/merged/fish_mapping_merged_production.csv',
                'images_dir': 'datasets/training/images',
                'img_size': 224,
                'batch_size': 16,  # Smaller batches for better learning
                'num_workers': 2
            },
            'model': {
                'type': 'efficientnet_b0',  # Smaller model
                'dropout': 0.5,
                'freeze_epochs': 2  # Freeze backbone initially
            },
            'training': {
                'epochs': 50,  # More epochs
                'learning_rate': 0.0001,  # Lower LR
                'weight_decay': 0.01,  # Stronger regularization
                'patience': 10,
                'min_images_per_class': 15,  # Higher minimum
                'use_class_weights': True,  # Handle imbalance
                'label_smoothing': 0.1  # Prevent overconfidence
            },
            'augmentation': {
                'enabled': True,
                'strong_aug': True,  # More aggressive augmentation
                'horizontal_flip': 0.5,
                'vertical_flip': 0.2,
                'rotation': 30,
                'brightness': 0.3,
                'contrast': 0.3,
                'saturation': 0.3,
                'hue': 0.1
            }
        }
        
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config
    
    def setup_directories(self):
        for d in ['training/checkpoints', 'training/logs', 'training/outputs']:
            Path(d).mkdir(parents=True, exist_ok=True)
    
    def get_improved_transforms(self, train=True):
        """Enhanced augmentation for better generalization"""
        img_size = self.config['data']['img_size']
        
        if train and self.config['augmentation']['enabled']:
            aug = self.config['augmentation']
            return transforms.Compose([
                transforms.Resize((img_size + 32, img_size + 32)),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(p=aug.get('horizontal_flip', 0.5)),
                transforms.RandomVerticalFlip(p=aug.get('vertical_flip', 0.2)),
                transforms.RandomRotation(aug.get('rotation', 30)),
                transforms.ColorJitter(
                    brightness=aug.get('brightness', 0.3),
                    contrast=aug.get('contrast', 0.3),
                    saturation=aug.get('saturation', 0.3),
                    hue=aug.get('hue', 0.1)
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2)
            ])
        else:
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def prepare_dataset(self):
        """Load and prepare dataset"""
        print("\n Loading images...")
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
            print(" No images found!")
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
        
        print(f" Dataset: {len(df)} images, {len(unique_species)} species")
        for species, count in counts[valid].items():
            print(f"   {species}: {count} images")
        
        # Save mapping
        with open('training/outputs/species_mapping.json', 'w') as f:
            json.dump({
                'species_to_id': self.species_to_id,
                'id_to_species': self.id_to_species
            }, f, indent=2)
        
        return df
    
    def create_balanced_loaders(self, df):
        """Create dataloaders with class balancing"""
        train_df, val_df = train_test_split(
            df, test_size=0.2, stratify=df['species_id'], random_state=42
        )
        
        print(f"\n Split: {len(train_df)} train, {len(val_df)} val")
        
        train_dataset = ImprovedFishDataset(
            train_df, self.config['data']['images_dir'],
            self.get_improved_transforms(train=True)
        )
        
        val_dataset = ImprovedFishDataset(
            val_df, self.config['data']['images_dir'],
            self.get_improved_transforms(train=False)
        )
        
        # Weighted sampling for class balance
        if self.config['training'].get('use_class_weights', True):
            class_counts = Counter(train_df['species_id'])
            weights = [1.0 / class_counts[sid] for sid in train_df['species_id']]
            sampler = WeightedRandomSampler(weights, len(weights))
            shuffle = False
        else:
            sampler = None
            shuffle = True
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['data']['batch_size'],
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
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
        """Initialize improved model"""
        print(f"\n Building improved model for {num_classes} classes...")
        
        self.model = ImprovedFishClassifier(
            num_classes=num_classes,
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        # Label smoothing for better calibration
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config['training'].get('label_smoothing', 0.1)
        )
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Optionally freeze backbone initially
        self.freeze_backbone()
        
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Trainable parameters: {params:,}")
    
    def freeze_backbone(self):
        """Freeze backbone for initial training"""
        for param in self.model.backbone.features.parameters():
            param.requires_grad = False
        print("    Backbone frozen for initial training")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.model.parameters():
            param.requires_grad = True
        print("    Backbone unfrozen for fine-tuning")
    
    def train_epoch(self, loader, epoch):
        """Train one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
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
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
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
        print(" IMPROVED TRAINING")
        print("="*70)
        
        df = self.prepare_dataset()
        if df is None:
            return
        
        train_loader, val_loader = self.create_balanced_loaders(df)
        self.initialize_model(len(self.species_to_id))
        
        best_val_acc = 0.0
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(self.config['training']['epochs']):
            # Unfreeze backbone after initial epochs
            if epoch == self.config['model'].get('freeze_epochs', 2):
                self.unfreeze_backbone()
                # Reduce learning rate when unfreezing
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config['training']['learning_rate'] / 10
            
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate(val_loader)
            
            self.scheduler.step()
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
                self.save_checkpoint('best_model.pth', epoch, val_acc)
                print(f"    New best! Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1
            
            if patience_counter >= self.config['training']['patience']:
                print(f"\n  Early stopping")
                break
        
        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print(" TRAINING COMPLETE!")
        print("="*70)
        print(f"Best Val Accuracy: {best_val_acc:.2f}%")
        print(f"Time: {elapsed/60:.1f} minutes")
        
        self.save_checkpoint('final_model.pth', epoch, val_acc)
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
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(self.history['train_acc'], label='Train')
        axes[1].plot(self.history['val_acc'], label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training/outputs/training_history_improved.png', dpi=300)
        print(f" Saved: training/outputs/training_history_improved.png")

if __name__ == "__main__":
    trainer = ImprovedTrainer()
    trainer.train()