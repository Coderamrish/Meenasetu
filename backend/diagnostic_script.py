"""
Model Loading Diagnostic Script
Run this to diagnose and fix model loading issues
"""

import json
from pathlib import Path
import sys

def check_disease_model_files():
    """Check disease model files and class mappings"""
    print("\n" + "="*80)
    print("🏥 DISEASE MODEL DIAGNOSTIC")
    print("="*80)
    
    base_dir = Path.cwd()
    if not (base_dir / "training").exists():
        base_dir = base_dir.parent
    
    checkpoints_dir = base_dir / "training" / "checkpoints"
    
    print(f"\nCheckpoints directory: {checkpoints_dir}")
    print(f"Exists: {checkpoints_dir.exists()}")
    
    if not checkpoints_dir.exists():
        print("❌ Checkpoints directory not found!")
        return
    
    # Check disease models
    disease_models = {
        "final.keras": "classes2.json",
        "s1.keras": "classes2.json"
    }
    
    for model_file, mapping_file in disease_models.items():
        print(f"\n{'='*80}")
        print(f"Checking: {model_file}")
        print(f"{'='*80}")
        
        model_path = checkpoints_dir / model_file
        mapping_path = checkpoints_dir / mapping_file
        
        print(f"Model path: {model_path}")
        print(f"Model exists: {model_path.exists()}")
        
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"Model size: {size_mb:.2f} MB")
        
        print(f"\nMapping path: {mapping_path}")
        print(f"Mapping exists: {mapping_path.exists()}")
        
        if mapping_path.exists():
            try:
                with open(mapping_path, 'r') as f:
                    mapping = json.load(f)
                
                print(f"\n📋 Mapping Contents:")
                print(f"Type: {type(mapping)}")
                print(f"Keys: {list(mapping.keys())[:10]}")
                
                # Analyze structure
                if isinstance(mapping, dict):
                    print(f"Number of entries: {len(mapping)}")
                    
                    # Check format
                    sample_key = list(mapping.keys())[0]
                    sample_value = mapping[sample_key]
                    
                    print(f"\nSample entry:")
                    print(f"  Key: {sample_key} (type: {type(sample_key)})")
                    print(f"  Value: {sample_value} (type: {type(sample_value)})")
                    
                    # Determine format
                    if all(str(k).isdigit() for k in mapping.keys()):
                        print(f"\n✅ Format: Numeric keys (id -> disease_name)")
                        print(f"   This will be converted to disease_name -> id")
                        
                        # Show converted format
                        converted = {v: int(k) for k, v in mapping.items()}
                        print(f"\n   Converted format (first 5):")
                        for i, (disease, idx) in enumerate(list(converted.items())[:5]):
                            print(f"     {disease}: {idx}")
                    
                    elif 'disease_to_id' in mapping:
                        print(f"\n✅ Format: Has 'disease_to_id' key")
                        print(f"   Direct format, no conversion needed")
                        disease_to_id = mapping['disease_to_id']
                        print(f"\n   Diseases (first 5):")
                        for i, (disease, idx) in enumerate(list(disease_to_id.items())[:5]):
                            print(f"     {disease}: {idx}")
                    
                    elif 'id_to_disease' in mapping:
                        print(f"\n✅ Format: Has 'id_to_disease' key")
                        print(f"   Will be converted to disease_to_id")
                        id_to_disease = mapping['id_to_disease']
                        converted = {v: int(k) for k, v in id_to_disease.items()}
                        print(f"\n   Converted format (first 5):")
                        for i, (disease, idx) in enumerate(list(converted.items())[:5]):
                            print(f"     {disease}: {idx}")
                    
                    else:
                        print(f"\n⚠️ Format: Unknown - trying to infer...")
                        all_values_are_ints = all(isinstance(v, int) for v in mapping.values())
                        if all_values_are_ints:
                            print(f"   Appears to be direct disease_to_id format")
                            print(f"\n   Diseases (first 5):")
                            for i, (disease, idx) in enumerate(list(mapping.items())[:5]):
                                print(f"     {disease}: {idx}")
                        else:
                            print(f"   ❌ Cannot determine format!")
                            print(f"   Full content: {mapping}")
                
            except Exception as e:
                print(f"❌ Error reading mapping: {e}")
                import traceback
                print(traceback.format_exc())

def check_species_models():
    """Check species classification models"""
    print("\n" + "="*80)
    print("🐟 SPECIES MODEL DIAGNOSTIC")
    print("="*80)
    
    base_dir = Path.cwd()
    if not (base_dir / "training").exists():
        base_dir = base_dir.parent
    
    checkpoints_dir = base_dir / "training" / "checkpoints"
    
    species_models = {
        "fish_model.pth": "class_mapping1.json",
        "best_model.pth": "class_mapping.json",
        "final_model.pth": "class_mapping.json"
    }
    
    for model_file, mapping_file in species_models.items():
        print(f"\n{'='*80}")
        print(f"Checking: {model_file}")
        
        model_path = checkpoints_dir / model_file
        mapping_path = checkpoints_dir / mapping_file
        
        print(f"Model exists: {model_path.exists()}")
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"Model size: {size_mb:.2f} MB")
        
        print(f"Mapping exists: {mapping_path.exists()}")

def suggest_fixes():
    """Suggest fixes for common issues"""
    print("\n" + "="*80)
    print("🔧 SUGGESTED FIXES")
    print("="*80)
    
    print("""
1. IF DISEASE MODELS FAIL TO LOAD:
   
   Check classes2.json format. It should be ONE of:
   
   Option A (Recommended):
   {
     "0": "disease_name_1",
     "1": "disease_name_2",
     ...
   }
   
   Option B:
   {
     "disease_to_id": {
       "disease_name_1": 0,
       "disease_name_2": 1,
       ...
     }
   }
   
   Option C:
   {
     "id_to_disease": {
       "0": "disease_name_1",
       "1": "disease_name_2",
       ...
     }
   }

2. IF CLASS COUNT MISMATCH:
   - Model output classes: Check model.output_shape[-1]
   - Mapping classes: Count entries in JSON
   - They MUST match!

3. IF TENSORFLOW NOT AVAILABLE:
   - Install: pip install tensorflow
   - Or for GPU: pip install tensorflow-gpu

4. IF MODELS ARE MISSING:
   - Check path: training/checkpoints/
   - Ensure .keras or .pth files exist
   - Check file permissions

5. REPLACE YOUR chain.py:
   - Copy the fixed version provided
   - Restart your FastAPI server
   - Run test: python test_production.py
    """)

def main():
    """Run full diagnostic"""
    print("\n" + "="*80)
    print("🔍 MEENASETU MODEL DIAGNOSTIC TOOL")
    print("="*80)
    
    check_disease_model_files()
    check_species_models()
    suggest_fixes()
    
    print("\n" + "="*80)
    print("✅ DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the output above")
    print("2. Fix any identified issues")
    print("3. Replace chain.py with the fixed version")
    print("4. Restart your server: uvicorn main:app --reload")
    print("5. Test: python test_production.py")

if __name__ == "__main__":
    main()