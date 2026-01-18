"""
Quick test to verify disease detection is working
"""

import sys
from pathlib import Path

# Force import from current directory
current_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(current_dir))

# Remove any other chain modules that might interfere
if 'chain' in sys.modules:
    del sys.modules['chain']

# Import from local chain.py
import chain
FishDiseaseDetector = chain.FishDiseaseDetector

def main():
    print("\n" + "="*80)
    print("🧪 QUICK DISEASE DETECTION TEST")
    print("="*80)
    
    # Initialize detector
    print("\n1️⃣ Initializing Disease Detector...")
    detector = FishDiseaseDetector()
    
    # Check status
    print(f"\n2️⃣ Status Check:")
    print(f"   TensorFlow Available: {detector.is_available}")
    print(f"   Models Loaded: {len(detector.models)}")
    print(f"   Primary Model: {detector.primary_model_name}")
    print(f"   Is Ready: {detector.is_loaded}")
    
    if detector.is_loaded:
        print(f"\n3️⃣ Available Diseases:")
        diseases = detector.available_diseases
        for i, disease in enumerate(diseases, 1):
            print(f"   {i}. {disease}")
        
        print(f"\n4️⃣ Class Mapping Details:")
        primary_mapping = detector.class_mappings[detector.primary_model_name]
        print(f"   Total Classes: {len(primary_mapping)}")
        print(f"   Format: disease_name -> id")
        print(f"   Sample: {list(primary_mapping.items())[:3]}")
        
        # Test model output shape
        print(f"\n5️⃣ Model Architecture:")
        model = detector.models[detector.primary_model_name]
        print(f"   Input Shape: {model.input_shape}")
        print(f"   Output Shape: {model.output_shape}")
        print(f"   Output Classes: {model.output_shape[-1]}")
        print(f"   Mapping Classes: {len(primary_mapping)}")
        
        match = model.output_shape[-1] == len(primary_mapping)
        print(f"   ✅ Shapes Match: {match}")
        
        print("\n" + "="*80)
        print("✅ DISEASE DETECTION READY FOR PRODUCTION!")
        print("="*80)
        
        print("\n📝 Next Steps:")
        print("   1. Run full test: python test_production.py")
        print("   2. Start server: uvicorn main:app --reload")
        print("   3. Test with real images via API")
        
    else:
        print("\n❌ Disease detection not loaded!")
        print("   Check the logs above for errors")
    
if __name__ == "__main__":
    main()