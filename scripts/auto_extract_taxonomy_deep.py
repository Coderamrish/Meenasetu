"""
MeenaSetu - Table Structure Inspector
Shows exactly what's in the WB Fish Diversity tables
"""

import pandas as pd
from pathlib import Path

def inspect_wb_tables():
    print("\n" + "="*70)
    print("🔍 WB Fish Diversity Table Inspector")
    print("="*70)
    
    processed_dir = Path("datasets/processed")
    
    tables = [
        'FreshwaterfishdiversityofWestBengal_table_1.csv',
        'FreshwaterfishdiversityofWestBengal_table_2.csv',
        'FreshwaterfishdiversityofWestBengal_table_3.csv',
    ]
    
    for table_name in tables:
        filepath = processed_dir / table_name
        
        if not filepath.exists():
            print(f"\n❌ {table_name} not found")
            continue
        
        print(f"\n{'='*70}")
        print(f"📄 {table_name}")
        print('='*70)
        
        df = pd.read_csv(filepath)
        
        print(f"\n📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\n📋 Columns:")
        for i, col in enumerate(df.columns):
            print(f"   {i+1}. '{col}'")
        
        print(f"\n🔍 First 10 rows:\n")
        print(df.head(10).to_string())
        
        print(f"\n🔍 Column '{df.columns[0]}' - First 20 values:")
        for i, val in enumerate(df[df.columns[0]].head(20)):
            print(f"   {i+1}. {val}")
    
    print("\n" + "="*70)
    print("✅ Inspection complete!")
    print("="*70)
    print("\n💡 Look for patterns like:")
    print("   • 'Order: Cypriniformes'")
    print("   • 'Family: Cyprinidae'")
    print("   • 'Genus species (Author, Year)'")

if __name__ == "__main__":
    inspect_wb_tables()