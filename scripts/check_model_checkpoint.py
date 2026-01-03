import torch
import json
import os

# Load the checkpoint
checkpoint_path = r"C:\Users\AMRISH\Documents\Meenasetu\training\checkpoints\best_model.pth"

print("=" * 70)
print("🔍 CHECKPOINT INSPECTION")
print("=" * 70)
print(f"\n📁 Loading: {checkpoint_path}")

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check what's inside the checkpoint
    if isinstance(checkpoint, dict):
        print("\n📦 Checkpoint Contents:")
        print(f"   Type: Dictionary with {len(checkpoint)} keys")
        print("\n🔑 Available Keys:")
        for i, key in enumerate(checkpoint.keys(), 1):
            value = checkpoint[key]
            if isinstance(value, dict):
                print(f"   {i}. {key} → dict with {len(value)} items")
            elif isinstance(value, list):
                print(f"   {i}. {key} → list with {len(value)} items")
            elif isinstance(value, (int, float, str)):
                print(f"   {i}. {key} → {value}")
            else:
                print(f"   {i}. {key} → {type(value).__name__}")
        
        # Check for class mapping
        print("\n" + "=" * 70)
        print("🏷️ CLASS MAPPING SEARCH")
        print("=" * 70)
        
        found_mapping = False
        
        if 'class_mapping' in checkpoint:
            print("\n✅ Found 'class_mapping'!")
            mapping = checkpoint['class_mapping']
            print(f"   Classes: {len(mapping)}")
            print("\n   Content:")
            for name, idx in list(mapping.items())[:10]:
                print(f"      {idx}: {name}")
            if len(mapping) > 10:
                print(f"      ... and {len(mapping) - 10} more")
            
            # Save it
            save_path = "class_mapping.json"
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Saved to: {save_path}")
            found_mapping = True
        
        if 'class_names' in checkpoint:
            print("\n✅ Found 'class_names'!")
            names = checkpoint['class_names']
            print(f"   Classes: {len(names)}")
            print("\n   Content:")
            for idx, name in enumerate(names[:10]):
                print(f"      {idx}: {name}")
            if len(names) > 10:
                print(f"      ... and {len(names) - 10} more")
            
            # Create mapping
            if not found_mapping:
                mapping = {name: idx for idx, name in enumerate(names)}
                save_path = "class_mapping.json"
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(mapping, f, indent=2, ensure_ascii=False)
                print(f"\n💾 Created and saved mapping to: {save_path}")
                found_mapping = True
        
        if 'idx_to_class' in checkpoint:
            print("\n✅ Found 'idx_to_class'!")
            idx_to_class = checkpoint['idx_to_class']
            print(f"   Classes: {len(idx_to_class)}")
            print("\n   Content:")
            for idx, name in list(idx_to_class.items())[:10]:
                print(f"      {idx}: {name}")
            if len(idx_to_class) > 10:
                print(f"      ... and {len(idx_to_class) - 10} more")
            
            # Create proper mapping
            if not found_mapping:
                mapping = {v: int(k) if isinstance(k, str) else k for k, v in idx_to_class.items()}
                save_path = "class_mapping.json"
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(mapping, f, indent=2, ensure_ascii=False)
                print(f"\n💾 Created and saved mapping to: {save_path}")
                found_mapping = True
        
        if 'classes' in checkpoint:
            print(f"\n📊 'classes' value: {checkpoint['classes']}")
        
        if not found_mapping:
            print("\n❌ No class mapping found in checkpoint!")
            print("   Searched for: 'class_mapping', 'class_names', 'idx_to_class'")
            print("\n💡 You'll need to create class_mapping.json manually")
        
        # Check model architecture
        print("\n" + "=" * 70)
        print("🏗️ MODEL ARCHITECTURE INFO")
        print("=" * 70)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("\n✅ Found 'model_state_dict'")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("\n✅ Found 'state_dict'")
        else:
            state_dict = checkpoint
            print("\n⚠️ Using checkpoint as state_dict directly")
        
        # Get output layer size
        num_classes = None
        for key in state_dict.keys():
            if 'fc.weight' in key:
                num_classes = state_dict[key].shape[0]
                print(f"\n🎯 Output layer size: {num_classes} classes")
                print(f"   (from key: {key})")
                break
        
        if num_classes is None:
            print("\n⚠️ Could not determine number of classes from model")
        
        # Training info
        print("\n" + "=" * 70)
        print("📈 TRAINING METADATA")
        print("=" * 70)
        
        if 'epoch' in checkpoint:
            print(f"\n   Epoch: {checkpoint['epoch']}")
        
        if 'best_acc' in checkpoint:
            print(f"   Best Accuracy: {checkpoint['best_acc']:.4f}")
        elif 'best_accuracy' in checkpoint:
            print(f"   Best Accuracy: {checkpoint['best_accuracy']:.4f}")
        
        if 'train_acc' in checkpoint:
            print(f"   Train Accuracy: {checkpoint['train_acc']:.4f}")
        
        if 'val_acc' in checkpoint:
            print(f"   Val Accuracy: {checkpoint['val_acc']:.4f}")
        
        if 'optimizer_state_dict' in checkpoint:
            print(f"   Optimizer: Saved ✓")
        
    else:
        print("\n⚠️ Checkpoint is state_dict only (not a dictionary)")
        print("   This means it only contains model weights, no metadata")
        
        # Try to get output size
        num_classes = None
        for key in checkpoint.keys():
            if 'fc.weight' in key:
                num_classes = checkpoint[key].shape[0]
                print(f"\n🎯 Output layer size: {num_classes} classes")
                break
        
        print("\n❌ No class mapping available - you'll need to create it manually")
    
    print("\n" + "=" * 70)
    print("✅ INSPECTION COMPLETE")
    print("=" * 70)

except Exception as e:
    print(f"\n❌ Error loading checkpoint: {e}")
    import traceback
    traceback.print_exc()