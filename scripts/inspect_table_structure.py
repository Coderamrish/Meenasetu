"""
MeenaSetu - Deep Taxonomy Extraction
Carefully extracts Order/Family/Species from WB Fish tables
"""

import pandas as pd
import re
from pathlib import Path
from collections import defaultdict

class DeepTaxonomyExtractor:
    def __init__(self):
        self.processed_dir = Path("datasets/processed")
        self.master_file = Path("data/final/fish_mapping_master.csv")
        self.taxonomy_map = defaultdict(dict)
        
    def load_master(self):
        """Load master dataset"""
        print("\n📖 Loading master dataset...")
        self.master_df = pd.read_csv(self.master_file)
        print(f"   ✅ {len(self.master_df)} species loaded\n")
    
    def parse_order_family_species(self, text):
        """Parse 'Order: X, Family: Y, Species: Z' format"""
        text = str(text).strip()
        
        order = None
        family = None
        species = None
        
        # Pattern 1: "Order: Cypriniformes"
        order_match = re.search(r'Order[:\s]+([A-Za-z]+)', text, re.IGNORECASE)
        if order_match:
            order = order_match.group(1).strip()
        
        # Pattern 2: "Family: Cyprinidae"
        family_match = re.search(r'Family[:\s]+([A-Za-z]+)', text, re.IGNORECASE)
        if family_match:
            family = family_match.group(1).strip()
        
        # Pattern 3: Species name (Genus species (Author, Year))
        species_match = re.search(r'([A-Z][a-z]+ [a-z]+(?:\s+\([^)]+\))?)', text)
        if species_match:
            species = species_match.group(1).strip()
        
        return order, family, species
    
    def extract_from_wb_tables(self):
        """Extract taxonomy from WB Fish Diversity tables"""
        print("🔍 Deep extraction from WB Fish Diversity tables...\n")
        
        # Target the main taxonomy tables
        target_files = [
            'FreshwaterfishdiversityofWestBengal_table_1.csv',
            'FreshwaterfishdiversityofWestBengal_table_2.csv',
            'FreshwaterfishdiversityofWestBengal_table_3.csv',
            'FreshwaterfishdiversityofWestBengal_table_4.csv',
            'FreshwaterfishdiversityofWestBengal_table_5.csv',
            'FreshwaterfishdiversityofWestBengal_table_6.csv',
        ]
        
        total_extracted = 0
        current_order = None
        current_family = None
        
        for filename in target_files:
            filepath = self.processed_dir / filename
            
            if not filepath.exists():
                continue
            
            print(f"📄 Processing {filename}...")
            
            try:
                df = pd.read_csv(filepath)
                
                # Look for the column with species data
                species_col = None
                for col in df.columns:
                    if 'species' in col.lower() or 'order' in col.lower():
                        species_col = col
                        break
                
                if not species_col:
                    # Try first column
                    species_col = df.columns[0]
                
                print(f"   Using column: '{species_col}'")
                
                extracted = 0
                for idx, row in df.iterrows():
                    cell_value = str(row[species_col]).strip()
                    
                    # Skip if NaN or empty
                    if cell_value == 'nan' or not cell_value:
                        continue
                    
                    # Check if this is an Order header
                    if 'Order:' in cell_value or cell_value.startswith('Order'):
                        order_match = re.search(r'Order[:\s]+([A-Za-z]+)', cell_value, re.IGNORECASE)
                        if order_match:
                            current_order = order_match.group(1).strip()
                            print(f"      📌 Order: {current_order}")
                        continue
                    
                    # Check if this is a Family header
                    if 'Family:' in cell_value or cell_value.startswith('Family'):
                        family_match = re.search(r'Family[:\s]+([A-Za-z]+)', cell_value, re.IGNORECASE)
                        if family_match:
                            current_family = family_match.group(1).strip()
                            print(f"         📌 Family: {current_family}")
                        continue
                    
                    # Extract species name
                    species_match = re.match(r'^([A-Z][a-z]+ [a-z]+)(?:\s+\([^)]+\))?', cell_value)
                    
                    if species_match:
                        species_name = species_match.group(1).strip()
                        
                        # Store taxonomy
                        if species_name not in self.taxonomy_map:
                            self.taxonomy_map[species_name] = {
                                'order': current_order,
                                'family': current_family
                            }
                            extracted += 1
                            total_extracted += 1
                
                print(f"   ✅ Extracted {extracted} species\n")
                
            except Exception as e:
                print(f"   ⚠️  Error: {e}\n")
        
        print(f"📊 Total taxonomy records extracted: {total_extracted}\n")
        return total_extracted
    
    def apply_to_master(self):
        """Apply extracted taxonomy to master dataset"""
        print("🔄 Applying taxonomy to master dataset...\n")
        
        updated_family = 0
        updated_order = 0
        
        for idx, row in self.master_df.iterrows():
            sci_name = row['scientific_name']
            
            if sci_name in self.taxonomy_map:
                taxonomy = self.taxonomy_map[sci_name]
                
                # Update family if missing
                if (pd.isna(row['family']) or str(row['family']).strip() == '') and taxonomy['family']:
                    self.master_df.at[idx, 'family'] = taxonomy['family']
                    updated_family += 1
                
                # Update order if missing
                if (pd.isna(row['order']) or str(row['order']).strip() == '') and taxonomy['order']:
                    self.master_df.at[idx, 'order'] = taxonomy['order']
                    updated_order += 1
        
        print(f"   ✅ Updated {updated_family} family entries")
        print(f"   ✅ Updated {updated_order} order entries\n")
        
        return updated_family, updated_order
    
    def show_coverage(self):
        """Show taxonomy coverage statistics"""
        print("📊 Current Taxonomy Coverage:\n")
        
        total = len(self.master_df)
        
        has_family = self.master_df['family'].notna().sum()
        has_order = self.master_df['order'].notna().sum()
        
        print(f"   Family: {has_family}/{total} ({has_family/total*100:.1f}%)")
        print(f"   Order:  {has_order}/{total} ({has_order/total*100:.1f}%)")
        
        # High quality count
        complete_fields = ['scientific_name', 'family', 'order', 'habitat', 'iucn_status']
        self.master_df['temp_score'] = 0
        for field in complete_fields:
            self.master_df['temp_score'] += self.master_df[field].notna().astype(int)
        
        high_quality = (self.master_df['temp_score'] >= 4).sum()
        print(f"\n   High Quality (≥4/5 core fields): {high_quality} ({high_quality/total*100:.1f}%)")
    
    def manual_extraction_guide(self):
        """Show species that need manual review"""
        print("\n🎯 Species Needing Manual Review:\n")
        
        missing_both = self.master_df[
            (self.master_df['family'].isna()) & 
            (self.master_df['order'].isna())
        ]
        
        if len(missing_both) > 0:
            print(f"   Found {len(missing_both)} species missing both family and order\n")
            print("   Sample (first 20):")
            for idx, row in missing_both.head(20).iterrows():
                print(f"      • {row['scientific_name']}")
            
            # Save full list
            output_file = Path("data/final/reports/needs_taxonomy.csv")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            missing_both[['scientific_name', 'local_name', 'habitat']].to_csv(
                output_file, index=False
            )
            print(f"\n   📝 Full list saved: {output_file}")
    
    def save_results(self):
        """Save updated master file"""
        print("\n💾 Saving updated data...")
        
        # Backup original
        backup_file = self.master_file.parent / f"{self.master_file.stem}_pre_deep_extract.csv"
        if self.master_file.exists():
            import shutil
            shutil.copy(self.master_file, backup_file)
            print(f"   ✅ Backup: {backup_file}")
        
        # Save updated
        self.master_df.to_csv(self.master_file, index=False)
        print(f"   ✅ Updated: {self.master_file}")
        
        # Save taxonomy map for reference
        taxonomy_df = pd.DataFrame([
            {'species': k, 'family': v['family'], 'order': v['order']}
            for k, v in self.taxonomy_map.items()
        ])
        
        taxonomy_file = Path("data/final/extracted_taxonomy_map.csv")
        taxonomy_df.to_csv(taxonomy_file, index=False)
        print(f"   ✅ Taxonomy map: {taxonomy_file}")
    
    def run(self):
        """Run complete deep extraction"""
        print("\n" + "="*70)
        print("🔬 MeenaSetu - Deep Taxonomy Extraction")
        print("="*70)
        print("\nThis will carefully re-extract Order/Family data from source tables")
        print("="*70)
        
        self.load_master()
        
        extracted = self.extract_from_wb_tables()
        
        if extracted > 0:
            family_count, order_count = self.apply_to_master()
            self.show_coverage()
            self.manual_extraction_guide()
            self.save_results()
            
            print("\n" + "="*70)
            print("✅ DEEP EXTRACTION COMPLETE!")
            print("="*70)
            
            print(f"\n📈 Improvements:")
            print(f"   • Extracted taxonomy for {extracted} species")
            print(f"   • Updated {family_count} family entries")
            print(f"   • Updated {order_count} order entries")
            
            print("\n🎯 Next Steps:")
            print("   1. Run validator to see overall improvement")
            print("   2. Review: data/final/reports/needs_taxonomy.csv")
            print("   3. Manually research species missing taxonomy")
            print("   4. Run enrichment script again to propagate changes")
        else:
            print("\n⚠️  No additional taxonomy data extracted")
            print("   The source tables may have a different structure")
            print("\n💡 Try manually inspecting:")
            print("   datasets/processed/FreshwaterfishdiversityofWestBengal_table_1.csv")

if __name__ == "__main__":
    extractor = DeepTaxonomyExtractor()
    extractor.run()