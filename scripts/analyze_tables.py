"""
Analyze extracted tables to find fish species data
Find the most useful tables for building fish_mapping.csv
"""

import pandas as pd
from pathlib import Path
import re

class TableAnalyzer:
    """Analyze extracted CSV tables to find fish data"""
    
    def __init__(self, processed_dir="datasets/processed"):
        self.processed_dir = Path(processed_dir)
        
    def analyze_all_tables(self):
        """Analyze all extracted CSV files"""
        print("🔍 Analyzing extracted tables...")
        print("=" * 70)
        
        csv_files = list(self.processed_dir.glob("*.csv"))
        print(f"\nFound {len(csv_files)} CSV files")
        
        # Priority files to check first
        priority_files = [
            "FreshwaterfishdiversityofWestBengal",  # Fish species list
            "CATLA_Pamphlet",  # Specific species info
            "AnnualReport_2016-17",  # CIFA data
            "Handbook"  # Statistics
        ]
        
        useful_tables = []
        
        # Check each priority source
        for priority in priority_files:
            matching_files = [f for f in csv_files if priority in f.name]
            
            if matching_files:
                print(f"\n📊 {priority}: {len(matching_files)} tables")
                
                for csv_file in matching_files[:5]:  # Check first 5 tables
                    try:
                        df = pd.read_csv(csv_file)
                        score = self.score_table_usefulness(df, csv_file.name)
                        
                        if score > 0:
                            useful_tables.append({
                                'file': csv_file.name,
                                'score': score,
                                'rows': len(df),
                                'columns': len(df.columns),
                                'sample_columns': list(df.columns[:5])
                            })
                    except Exception as e:
                        continue
        
        # Sort by usefulness score
        useful_tables.sort(key=lambda x: x['score'], reverse=True)
        
        # Display top 20 most useful tables
        print("\n" + "=" * 70)
        print("🎯 TOP 20 MOST USEFUL TABLES FOR FISH MAPPING")
        print("=" * 70)
        
        for i, table in enumerate(useful_tables[:20], 1):
            print(f"\n{i}. {table['file']}")
            print(f"   Score: {table['score']} | Rows: {table['rows']} | Cols: {table['columns']}")
            print(f"   Columns: {table['sample_columns']}")
        
        return useful_tables
    
    def score_table_usefulness(self, df, filename):
        """Score how useful a table is for fish mapping"""
        score = 0
        
        # Get column names as lowercase
        col_str = ' '.join(df.columns).lower()
        
        # Check for fish-related keywords
        fish_keywords = [
            'species', 'fish', 'name', 'scientific',
            'labeo', 'catla', 'rohu', 'carp', 'tilapia'
        ]
        
        for keyword in fish_keywords:
            if keyword in col_str:
                score += 10
        
        # Check for parameter keywords
        param_keywords = [
            'temperature', 'temp', 'ph', 'density',
            'stocking', 'growth', 'culture', 'production'
        ]
        
        for keyword in param_keywords:
            if keyword in col_str:
                score += 5
        
        # Bonus for reasonable table size
        if 5 <= len(df) <= 500:  # Not too small or large
            score += 5
        
        if 3 <= len(df.columns) <= 15:  # Good number of columns
            score += 5
        
        # Check actual data for fish names
        data_str = ' '.join(df.astype(str).values.flatten()[:100]).lower()
        
        common_fish = [
            'rohu', 'catla', 'mrigal', 'tilapia', 'carp',
            'labeo', 'cirrhinus', 'oreochromis', 'pangasius'
        ]
        
        for fish in common_fish:
            if fish in data_str:
                score += 15
                break  # Only count once
        
        return score
    
    def extract_wb_fish_diversity(self):
        """Extract from WB Fish Diversity tables - these are GOLD"""
        print("\n" + "=" * 70)
        print("🐟 EXTRACTING: West Bengal Fish Diversity Tables")
        print("=" * 70)
        
        wb_tables = list(self.processed_dir.glob("FreshwaterfishdiversityofWestBengal_table_*.csv"))
        
        all_species = []
        
        for table_file in wb_tables:
            try:
                df = pd.read_csv(table_file)
                print(f"\n📄 {table_file.name}")
                print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
                print(f"   Columns: {list(df.columns)}")
                
                # This likely has species lists!
                # Look for columns with fish names
                for col in df.columns:
                    if 'species' in col.lower() or 'name' in col.lower():
                        species_list = df[col].dropna().tolist()
                        print(f"   Found {len(species_list)} entries in '{col}'")
                        all_species.extend(species_list[:10])  # Sample
                
                # Show first few rows
                print("\nSample data:")
                print(df.head(3).to_string())
                
            except Exception as e:
                print(f"   Error: {e}")
        
        print(f"\n✅ Total species mentions found: {len(all_species)}")
        return all_species
    
    def create_starter_dataset(self):
        """Create starter dataset from extracted tables"""
        print("\n" + "=" * 70)
        print("🏗️ CREATING STARTER DATASET")
        print("=" * 70)
        
        # Try to find the best table with fish data
        best_table = None
        best_score = 0
        
        for csv_file in self.processed_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                score = self.score_table_usefulness(df, csv_file.name)
                
                if score > best_score:
                    best_score = score
                    best_table = (csv_file, df)
            except:
                continue
        
        if best_table:
            filename, df = best_table
            print(f"\n✅ Best table found: {filename.name}")
            print(f"   Score: {best_score}")
            print(f"   Shape: {df.shape}")
            print("\nColumns:")
            print(df.columns.tolist())
            print("\nFirst 5 rows:")
            print(df.head().to_string())
            
            # Save as starting point
            output_path = Path("data/final/extracted_fish_data.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"\n💾 Saved to: {output_path}")
        
        return best_table


def main():
    """Main execution"""
    print("\n" + "=" * 70)
    print("🐟 MeenaSetu - Table Analyzer")
    print("=" * 70)
    
    analyzer = TableAnalyzer()
    
    # Step 1: Analyze all tables
    useful_tables = analyzer.analyze_all_tables()
    
    # Step 2: Deep dive into WB Fish Diversity
    analyzer.extract_wb_fish_diversity()
    
    # Step 3: Create starter dataset
    analyzer.create_starter_dataset()
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    
    print("\n📋 Next Steps:")
    print("   1. Review the top 20 tables listed above")
    print("   2. Open promising tables in Excel/VS Code")
    print("   3. Manually extract fish data to data/final/fish_mapping.csv")
    print("   4. Focus on WB Fish Diversity tables first (they have 267 species!)")
    
    print("\n💡 Quick Commands:")
    print("   # View a specific table")
    print("   code datasets/processed/FreshwaterfishdiversityofWestBengal_table_0.csv")
    print("\n   # Open in Excel")
    print("   start datasets/processed/FreshwaterfishdiversityofWestBengal_table_0.csv")


if __name__ == "__main__":
    main()