import torch
import json
import os

print("=" * 70)
print("🐟 EXTRACTING SPECIES MAPPING")
print("=" * 70)

# Load checkpoint
checkpoint_path = r"C:\Users\AMRISH\Documents\Meenasetu\training\checkpoints\best_model.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("\n📦 Checkpoint 'species_mapping' content:")
species_mapping = checkpoint['species_mapping']
print(json.dumps(species_mapping, indent=2, ensure_ascii=False))

# Also check the config
print("\n⚙️ Checkpoint 'config' content:")
config = checkpoint['config']
print(json.dumps(config, indent=2, ensure_ascii=False))

# Check the external files
print("\n" + "=" * 70)
print("📄 EXTERNAL MAPPING FILES")
print("=" * 70)

files_to_check = [
    r"C:\Users\AMRISH\Documents\Meenasetu\training\outputs\species_mapping.json",
    r"C:\Users\AMRISH\Documents\Meenasetu\training\outputs\species_mapping_advanced.json"
]

for filepath in files_to_check:
    if os.path.exists(filepath):
        print(f"\n📄 {os.path.basename(filepath)}:")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"   Keys: {list(data.keys())}")
        
        if 'species_to_id' in data:
            print(f"\n   🔢 species_to_id ({len(data['species_to_id'])} classes):")
            for species, idx in list(data['species_to_id'].items())[:15]:
                print(f"      {idx}: {species}")
            if len(data['species_to_id']) > 15:
                print(f"      ... and {len(data['species_to_id']) - 15} more")
        
        if 'id_to_species' in data:
            print(f"\n   🐠 id_to_species ({len(data['id_to_species'])} classes):")
            for idx, species in list(data['id_to_species'].items())[:15]:
                print(f"      {idx}: {species}")
            if len(data['id_to_species']) > 15:
                print(f"      ... and {len(data['id_to_species']) - 15} more")

# Now create the proper class_mapping.json for the vector DB script
print("\n" + "=" * 70)
print("💾 CREATING class_mapping.json")
print("=" * 70)

# Try to get the best mapping
class_mapping = None

# First, try the external file (most likely to be correct)
if os.path.exists(files_to_check[0]):
    with open(files_to_check[0], 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'species_to_id' in data:
        class_mapping = data['species_to_id']
        print(f"\n✅ Using species_to_id from {os.path.basename(files_to_check[0])}")
        print(f"   Total classes: {len(class_mapping)}")

# If that didn't work, try the checkpoint
if class_mapping is None and 'species_to_id' in species_mapping:
    class_mapping = species_mapping['species_to_id']
    print(f"\n✅ Using species_to_id from checkpoint")
    print(f"   Total classes: {len(class_mapping)}")

# If still nothing, try id_to_species and reverse it
if class_mapping is None:
    if os.path.exists(files_to_check[0]):
        with open(files_to_check[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'id_to_species' in data:
            id_to_species = data['id_to_species']
            class_mapping = {species: int(idx) for idx, species in id_to_species.items()}
            print(f"\n✅ Created mapping by reversing id_to_species")
            print(f"   Total classes: {len(class_mapping)}")

if class_mapping:
    # Save to multiple locations for convenience
    save_locations = [
        r"C:\Users\AMRISH\Documents\Meenasetu\training\checkpoints\class_mapping.json",
        r"C:\Users\AMRISH\Documents\Meenasetu\class_mapping.json"
    ]
    
    for save_path in save_locations:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(class_mapping, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved to: {save_path}")
    
    print("\n📊 Sample of class_mapping.json:")
    for species, idx in list(class_mapping.items())[:10]:
        print(f"   {idx}: {species}")
    if len(class_mapping) > 10:
        print(f"   ... and {len(class_mapping) - 10} more")
    
    print("\n✅ class_mapping.json created successfully!")
else:
    print("\n❌ Could not create class_mapping.json")
    print("   Please check the mapping files manually")

print("\n" + "=" * 70)