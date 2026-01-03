import os
import json
import glob

print("=" * 70)
print("🔍 SEARCHING FOR CLASS NAMES")
print("=" * 70)

# Search locations
search_locations = [
    r"C:\Users\AMRISH\Documents\Meenasetu\training",
    r"C:\Users\AMRISH\Documents\Meenasetu\training\logs",
    r"C:\Users\AMRISH\Documents\Meenasetu\training\checkpoints",
    r"C:\Users\AMRISH\Documents\Meenasetu\training\outputs",
    r"C:\Users\AMRISH\Documents\Meenasetu\datasets",
]

found_files = []

print("\n🔎 Searching for relevant files...\n")

for location in search_locations:
    if not os.path.exists(location):
        continue
    
    print(f"📂 Checking: {location}")
    
    # Look for JSON files
    json_files = glob.glob(os.path.join(location, "*.json"))
    for json_file in json_files:
        filename = os.path.basename(json_file)
        if any(keyword in filename.lower() for keyword in ['class', 'label', 'mapping', 'species', 'categories']):
            found_files.append(json_file)
            print(f"   ✅ Found: {filename}")
    
    # Look for text files with relevant names
    txt_files = glob.glob(os.path.join(location, "*.txt"))
    for txt_file in txt_files:
        filename = os.path.basename(txt_file)
        if any(keyword in filename.lower() for keyword in ['class', 'label', 'species', 'categories']):
            found_files.append(txt_file)
            print(f"   ✅ Found: {filename}")
    
    # Look in subdirectories
    for root, dirs, files in os.walk(location):
        if root == location:
            continue
        for file in files:
            filename = file.lower()
            if (filename.endswith('.json') or filename.endswith('.txt')) and \
               any(keyword in filename for keyword in ['class', 'label', 'mapping', 'species', 'categories']):
                found_files.append(os.path.join(root, file))
                print(f"   ✅ Found: {os.path.relpath(os.path.join(root, file), location)}")

print("\n" + "=" * 70)
print("📄 EXAMINING FILES")
print("=" * 70)

if not found_files:
    print("\n❌ No relevant files found!")
else:
    for filepath in found_files:
        print(f"\n📄 {os.path.basename(filepath)}")
        print(f"   Path: {filepath}")
        
        try:
            if filepath.endswith('.json'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"   Type: JSON")
                
                if isinstance(data, dict):
                    print(f"   Keys: {list(data.keys())[:5]}")
                    
                    # Check if it looks like a class mapping
                    if len(data) <= 50 and all(isinstance(v, int) for v in data.values()):
                        print(f"\n   🎯 Looks like a class mapping! ({len(data)} classes)")
                        print("\n   Sample entries:")
                        for name, idx in list(data.items())[:5]:
                            print(f"      {idx}: {name}")
                        
                        # Ask if we should use this
                        print(f"\n   💡 This could be your class mapping!")
                
                elif isinstance(data, list):
                    print(f"   Length: {len(data)}")
                    if len(data) <= 50:
                        print(f"\n   🎯 Looks like a class names list! ({len(data)} classes)")
                        print("\n   Sample entries:")
                        for idx, name in enumerate(data[:5]):
                            print(f"      {idx}: {name}")
            
            elif filepath.endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:20]  # First 20 lines
                
                print(f"   Type: Text file")
                print(f"   Lines: {len(lines)}")
                print("\n   Preview:")
                for line in lines[:10]:
                    print(f"      {line.strip()}")
        
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")

print("\n" + "=" * 70)
print("🔍 CHECKING DATASET STRUCTURE")
print("=" * 70)

# Check if there's a dataset with folder structure (each folder = class)
dataset_paths = [
    r"C:\Users\AMRISH\Documents\Meenasetu\datasets",
    r"C:\Users\AMRISH\Documents\Meenasetu\data",
]

for dataset_path in dataset_paths:
    if not os.path.exists(dataset_path):
        continue
    
    print(f"\n📂 Checking: {dataset_path}")
    
    # Look for image folders
    subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    # Check if subdirectories contain images
    image_folders = []
    for subdir in subdirs:
        subdir_path = os.path.join(dataset_path, subdir)
        files = os.listdir(subdir_path)
        
        # Check if it contains images
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
        if image_files:
            image_folders.append((subdir, len(image_files)))
    
    if image_folders:
        print(f"\n   🎯 Found {len(image_folders)} image folders (potential classes):")
        for folder, count in image_folders[:10]:
            print(f"      • {folder} ({count} images)")
        
        if len(image_folders) > 10:
            print(f"      ... and {len(image_folders) - 10} more folders")
        
        # Create class mapping from folder names
        print(f"\n   💡 Creating class mapping from folder names...")
        class_mapping = {folder: idx for idx, (folder, _) in enumerate(sorted(image_folders))}
        
        save_path = "class_mapping_from_folders.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(class_mapping, f, indent=2, ensure_ascii=False)
        
        print(f"   💾 Saved to: {save_path}")

print("\n" + "=" * 70)
print("✅ SEARCH COMPLETE")
print("=" * 70)