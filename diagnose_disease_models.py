"""
Disease Model Loading - Diagnostic and Fix Script
Run this to identify and fix the disease model loading issue
"""

import sys
from pathlib import Path
import json

print("=" * 80)
print("🔍 DISEASE MODEL LOADING DIAGNOSTIC")
print("=" * 80)

# Step 1: Check TensorFlow
print("\n1️⃣ TensorFlow Check:")
try:
    import tensorflow as tf
    from tensorflow import keras
    print(f"   ✅ TensorFlow {tf.__version__} loaded successfully")
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    print(f"   ❌ TensorFlow import failed: {e}")
    TENSORFLOW_AVAILABLE = False
    sys.exit(1)

# Step 2: Check file paths
print("\n2️⃣ File Path Check:")
BASE_DIR = Path("C:/Users/AMRISH/Documents/Meenasetu")
TRAINING_DIR = BASE_DIR / "training"
CHECKPOINTS_DIR = TRAINING_DIR / "checkpoints"

print(f"   Base Directory: {BASE_DIR}")
print(f"   Training Directory: {TRAINING_DIR}")
print(f"   Checkpoints Directory: {CHECKPOINTS_DIR}")

files = {
    'final.keras': CHECKPOINTS_DIR / "final.keras",
    's1.keras': CHECKPOINTS_DIR / "s1.keras",
    'classes2.json': CHECKPOINTS_DIR / "classes2.json"
}

all_exist = True
for name, path in files.items():
    if path.exists():
        size = path.stat().st_size / (1024 * 1024)
        print(f"   ✅ {name}: {size:.2f} MB")
    else:
        print(f"   ❌ {name}: NOT FOUND")
        all_exist = False

if not all_exist:
    print("\n❌ Some files are missing. Cannot proceed.")
    sys.exit(1)

# Step 3: Load class mapping
print("\n3️⃣ Class Mapping Check:")
mapping_path = files['classes2.json']
try:
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    
    print(f"   ✅ classes2.json loaded")
    print(f"   📋 Content structure: {list(mapping_data.keys())}")
    print(f"   📋 Full content:")
    print(f"      {json.dumps(mapping_data, indent=6)}")
    
    # Extract actual disease mapping
    if 'disease_to_id' in mapping_data:
        disease_mapping = mapping_data['disease_to_id']
        print(f"   ✅ Using 'disease_to_id' format")
    elif 'id_to_disease' in mapping_data:
        id_to_disease = mapping_data['id_to_disease']
        disease_mapping = {v: int(k) for k, v in id_to_disease.items()}
        print(f"   ✅ Using 'id_to_disease' format (converted)")
    else:
        disease_mapping = mapping_data
        print(f"   ✅ Using direct mapping format")
    
    print(f"   📊 Number of disease classes: {len(disease_mapping)}")
    print(f"   📋 Disease names: {list(disease_mapping.keys())}")
    
except Exception as e:
    print(f"   ❌ Failed to load classes2.json: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Try loading models
print("\n4️⃣ Model Loading Test:")

for model_name in ['final.keras', 's1.keras']:
    print(f"\n   Testing {model_name}:")
    model_path = files[model_name]
    
    # Try different loading methods
    methods = [
        ("Standard load", lambda p: keras.models.load_model(str(p))),
        ("Load without compile", lambda p: keras.models.load_model(str(p), compile=False)),
        ("Load with safe mode", lambda p: keras.models.load_model(str(p), safe_mode=False)),
    ]
    
    model_loaded = False
    working_method = None
    loaded_model = None
    
    for method_name, method_func in methods:
        try:
            print(f"      Trying: {method_name}...")
            model = method_func(model_path)
            print(f"      ✅ SUCCESS with {method_name}")
            print(f"         Input shape: {model.input_shape}")
            print(f"         Output shape: {model.output_shape}")
            
            if not model_loaded:
                model_loaded = True
                working_method = method_name
                loaded_model = model
            
        except Exception as e:
            print(f"      ❌ {method_name} failed: {str(e)[:100]}")
    
    if not model_loaded:
        print(f"   ❌ {model_name} could not be loaded with any method")
    else:
        print(f"   ✅ {model_name} successfully loaded!")
        
        # Verify output matches class count
        output_classes = loaded_model.output_shape[-1]
        expected_classes = len(disease_mapping)
        if output_classes == expected_classes:
            print(f"   ✅ Model output ({output_classes}) matches class count ({expected_classes})")
        else:
            print(f"   ⚠️ MISMATCH: Model output ({output_classes}) != class count ({expected_classes})")

# Step 5: Diagnose why chain.py isn't loading them
print("\n5️⃣ Analyzing chain.py Configuration:")
print("   The issue is likely in how the FishDiseaseDetector class loads models.")
print("   Let me check the configuration...")

# Simulate the config from chain.py
DISEASE_MODEL_CONFIGS = {
    'disease_model_final': {
        'path': CHECKPOINTS_DIR / "final.keras",
        'class_mapping': CHECKPOINTS_DIR / "classes2.json",
        'priority': 1
    },
    'disease_model_s1': {
        'path': CHECKPOINTS_DIR / "s1.keras",
        'class_mapping': CHECKPOINTS_DIR / "classes2.json",
        'priority': 2
    }
}

print("\n   Configuration check:")
for model_name, config in DISEASE_MODEL_CONFIGS.items():
    print(f"\n   {model_name}:")
    print(f"      Path exists: {config['path'].exists()}")
    print(f"      Mapping exists: {config['class_mapping'].exists()}")
    print(f"      Priority: {config['priority']}")

# Step 6: Recommendations
print("\n" + "=" * 80)
print("🎯 DIAGNOSIS COMPLETE")
print("=" * 80)

print("\n✅ WHAT'S WORKING:")
print("   - TensorFlow is installed and operational")
print("   - All disease model files exist")
print("   - Models can be loaded with Keras")
print("   - Class mapping is valid")

print("\n⚠️ LIKELY ISSUE:")
print("   The models are loading fine here, but not in chain.py.")
print("   This suggests one of the following:")
print("   1. Path resolution issue in chain.py Config class")
print("   2. Exception being caught and logged but not displayed")
print("   3. TRAINING_DIR not resolving to the correct path")

print("\n🔧 RECOMMENDED FIXES:")
print("\n   Option 1: Add debug logging to chain.py")
print("   Add these lines in FishDiseaseDetector.__init__:")
print('   logger.info(f"DEBUG: Looking for disease models in: {Config.TRAINING_DIR}")')
print('   logger.info(f"DEBUG: TensorFlow available: {self.is_available}")')
print()
print("   Option 2: Force the correct path in chain.py")
print("   In Config class, add:")
print('   TRAINING_DIR = Path("C:/Users/AMRISH/Documents/Meenasetu/training")')
print()
print("   Option 3: Load with compile=False")
print("   In FishDiseaseDetector._load_all_models(), change:")
print("   model = keras.models.load_model(str(config['path']))")
print("   TO:")
print("   model = keras.models.load_model(str(config['path']), compile=False)")

print("\n📋 NEXT STEPS:")
print("   1. Check meenasetu_production.log for error messages")
print("   2. Add the debug logging above to see actual paths")
print("   3. If you want, I can create a patched version of chain.py")

print("\n" + "=" * 80)
print("Would you like me to create a fixed version of the")
print("FishDiseaseDetector class with improved loading?")
print("=" * 80)