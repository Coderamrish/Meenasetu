"""
MeenaSetu - Automated Fish Species Extraction
Extracts 300+ species from your CSV tables automatically
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import re

class FishDataExtractor:
    def __init__(self):
        self.all_species = []
        self.processed_dir = "datasets/processed"
        
    def clean_column_name(self, col):
        """Clean column names by removing newlines and extra spaces"""
        if isinstance(col, str):
            return col.replace('\n', ' ').strip()
        return col
    
    def extract_scientific_name(self, text):
        """Extract scientific name from text"""
        if pd.isna(text):
            return None
        
        # Pattern: Genus species (Author, Year)
        pattern = r'([A-Z][a-z]+\s+[a-z]+)\s*\([^)]+\)'
        match = re.search(pattern, str(text))
        if match:
            return match.group(1)
        
        # Try simpler pattern: Genus species
        pattern2 = r'([A-Z][a-z]+\s+[a-z]+)'
        match2 = re.search(pattern2, str(text))
        if match2:
            return match2.group(1)
        
        return None
    
    def extract_size(self, text):
        """Extract size from text (e.g., '25.0 cm')"""
        if pd.isna(text):
            return None
        pattern = r'(\d+\.?\d*)\s*(cm|mm)'
        match = re.search(pattern, str(text))
        if match:
            return f"{match.group(1)} {match.group(2)}"
        return None
    
    def process_wb_table_1(self):
        """Process main species table from WB Fish Diversity"""
        file_path = f"{self.processed_dir}/FreshwaterfishdiversityofWestBengal_table_1.csv"
        try:
            df = pd.read_csv(file_path)
            print(f"📊 Processing {file_path}")
            print(f"   Columns: {list(df.columns)}")
            
            # Clean column names
            df.columns = [self.clean_column_name(col) for col in df.columns]
            
            for idx, row in df.iterrows():
                # Try to find scientific name in first column
                scientific_name = None
                for col in df.columns:
                    if scientific_name is None:
                        scientific_name = self.extract_scientific_name(row[col])
                        if scientific_name:
                            break
                
                if scientific_name and 'Family' not in scientific_name and 'Order' not in scientific_name:
                    species_data = {
                        'scientific_name': scientific_name,
                        'local_name': row.get('Local Name', ''),
                        'common_name': '',
                        'family': '',
                        'order': '',
                        'max_size': self.extract_size(row.get('Maximum size (TL)', '')),
                        'habitat': row.get('Environment', 'Freshwater'),
                        'iucn_status': row.get('IUCN Status', 'LC'),
                        'human_use': row.get('Human Use', ''),
                        'source': 'WB_Fish_Diversity_table_1'
                    }
                    self.all_species.append(species_data)
                    
            print(f"   ✅ Extracted {len([s for s in self.all_species if s['source'] == 'WB_Fish_Diversity_table_1'])} species")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    def process_wb_table_2_to_6(self):
        """Process tables 2-6 which have similar structure"""
        for table_num in range(2, 7):
            file_path = f"{self.processed_dir}/FreshwaterfishdiversityofWestBengal_table_{table_num}.csv"
            try:
                df = pd.read_csv(file_path)
                print(f"📊 Processing table_{table_num}")
                
                # Clean column names
                df.columns = [self.clean_column_name(col) for col in df.columns]
                
                # First column usually has scientific name
                first_col = df.columns[0]
                
                for idx, row in df.iterrows():
                    scientific_name = self.extract_scientific_name(row[first_col])
                    
                    if scientific_name and 'Family' not in scientific_name and 'Order' not in scientific_name:
                        # Try to map columns intelligently
                        local_name = row[df.columns[1]] if len(df.columns) > 1 else ''
                        habitat = row[df.columns[2]] if len(df.columns) > 2 else 'Freshwater'
                        max_size = self.extract_size(row[df.columns[3]]) if len(df.columns) > 3 else ''
                        human_use = row[df.columns[4]] if len(df.columns) > 4 else ''
                        iucn_status = row[df.columns[5]] if len(df.columns) > 5 else 'LC'
                        
                        species_data = {
                            'scientific_name': scientific_name,
                            'local_name': local_name if not pd.isna(local_name) else '',
                            'common_name': '',
                            'family': '',
                            'order': '',
                            'max_size': max_size if max_size else '',
                            'habitat': habitat if not pd.isna(habitat) else 'Freshwater',
                            'iucn_status': iucn_status if not pd.isna(iucn_status) else 'LC',
                            'human_use': human_use if not pd.isna(human_use) else '',
                            'source': f'WB_Fish_Diversity_table_{table_num}'
                        }
                        self.all_species.append(species_data)
                        
                print(f"   ✅ Extracted {len([s for s in self.all_species if s['source'] == f'WB_Fish_Diversity_table_{table_num}'])} species")
            except Exception as e:
                print(f"   ❌ Error processing table {table_num}: {e}")
    
    def search_all_tables_for_species(self):
        """Search through all CSV files for additional species"""
        print("\n🔍 Searching all tables for additional species...")
        
        processed_files = list(Path(self.processed_dir).glob("*.csv"))
        
        for file_path in processed_files:
            # Skip already processed files
            if 'FreshwaterfishdiversityofWestBengal_table_' in str(file_path):
                table_num = str(file_path).split('_table_')[1].split('.')[0]
                if table_num.isdigit() and int(table_num) <= 6:
                    continue
            
            try:
                df = pd.read_csv(file_path)
                
                # Look for scientific names in all cells
                for idx, row in df.iterrows():
                    for col in df.columns:
                        cell_value = row[col]
                        scientific_name = self.extract_scientific_name(cell_value)
                        
                        if scientific_name:
                            # Check if already added
                            if not any(s['scientific_name'] == scientific_name for s in self.all_species):
                                species_data = {
                                    'scientific_name': scientific_name,
                                    'local_name': '',
                                    'common_name': '',
                                    'family': '',
                                    'order': '',
                                    'max_size': '',
                                    'habitat': 'Freshwater',
                                    'iucn_status': 'LC',
                                    'human_use': '',
                                    'source': file_path.name
                                }
                                self.all_species.append(species_data)
            except Exception as e:
                pass  # Skip problematic files
    
    def remove_duplicates(self):
        """Remove duplicate species entries"""
        seen = set()
        unique_species = []
        
        for species in self.all_species:
            if species['scientific_name'] not in seen:
                seen.add(species['scientific_name'])
                unique_species.append(species)
        
        self.all_species = unique_species
        print(f"\n🧹 Removed duplicates. Unique species: {len(self.all_species)}")
    
    def enrich_with_family_order(self):
        """Try to enrich species with family and order information"""
        print("\n🔬 Enriching with family/order data...")
        
        # Common fish families and orders mapping
        family_order_map = {
            'Cyprinidae': 'Cypriniformes',
            'Bagridae': 'Siluriformes',
            'Siluridae': 'Siluriformes',
            'Channidae': 'Channiformes',
            'Anabantidae': 'Anabantiformes',
            'Osphronemidae': 'Anabantiformes',
            'Badidae': 'Perciformes',
            'Ambassidae': 'Perciformes',
            'Gobiidae': 'Gobiiformes',
            'Mastacembelidae': 'Synbranchiformes',
            'Cobitidae': 'Cypriniformes',
            'Balitoridae': 'Cypriniformes',
            'Schilbeidae': 'Siluriformes',
            'Heteropneustidae': 'Siluriformes',
            'Clariidae': 'Siluriformes',
            'Notopteridae': 'Osteoglossiformes',
            'Belonidae': 'Beloniformes',
            'Ambassidae': 'Perciformes'
        }
        
        # Genus to family mapping (common genera)
        genus_family_map = {
            'Labeo': 'Cyprinidae',
            'Catla': 'Cyprinidae',
            'Cirrhinus': 'Cyprinidae',
            'Puntius': 'Cyprinidae',
            'Pethia': 'Cyprinidae',
            'Barilius': 'Cyprinidae',
            'Mystus': 'Bagridae',
            'Rita': 'Bagridae',
            'Sperata': 'Bagridae',
            'Wallago': 'Siluridae',
            'Ompok': 'Siluridae',
            'Channa': 'Channidae',
            'Anabas': 'Anabantidae',
            'Trichogaster': 'Osphronemidae',
            'Badis': 'Badidae',
            'Heteropneustes': 'Heteropneustidae',
            'Clarias': 'Clariidae',
            'Notopterus': 'Notopteridae',
            'Chitala': 'Notopteridae',
            'Macrognathus': 'Mastacembelidae',
            'Mastacembelus': 'Mastacembelidae',
            'Lepidocephalichthys': 'Cobitidae',
            'Botia': 'Botiidae',
            'Glossogobius': 'Gobiidae'
        }
        
        for species in self.all_species:
            if not species['family']:
                # Extract genus
                genus = species['scientific_name'].split()[0]
                if genus in genus_family_map:
                    species['family'] = genus_family_map[genus]
            
            if not species['order'] and species['family']:
                if species['family'] in family_order_map:
                    species['order'] = family_order_map[species['family']]
        
        print(f"   ✅ Enriched {len([s for s in self.all_species if s['family']])} with family")
        print(f"   ✅ Enriched {len([s for s in self.all_species if s['order']])} with order")
    
    def save_to_csv(self, output_path="data/final/fish_mapping_auto_generated.csv"):
        """Save extracted species to CSV"""
        df = pd.DataFrame(self.all_species)
        
        # Reorder columns
        columns_order = [
            'scientific_name', 'common_name', 'local_name', 'family', 'order',
            'max_size', 'habitat', 'iucn_status', 'human_use', 'source'
        ]
        df = df[columns_order]
        
        # Create directory if not exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"\n💾 Saved to: {output_path}")
        print(f"   Total species: {len(df)}")
        
        return df
    
    def run(self):
        """Run the complete extraction pipeline"""
        print("\n" + "="*70)
        print("🐟 AUTOMATED FISH SPECIES EXTRACTION")
        print("="*70)
        
        # Step 1: Process main WB tables
        print("\n📖 Step 1: Processing WB Fish Diversity Tables...")
        self.process_wb_table_1()
        self.process_wb_table_2_to_6()
        
        # Step 2: Search all other tables
        print("\n📖 Step 2: Searching all tables for additional species...")
        self.search_all_tables_for_species()
        
        # Step 3: Remove duplicates
        self.remove_duplicates()
        
        # Step 4: Enrich with family/order
        self.enrich_with_family_order()
        
        # Step 5: Save to CSV
        df = self.save_to_csv()
        
        print("\n" + "="*70)
        print("✅ EXTRACTION COMPLETE!")
        print("="*70)
        
        # Print summary
        print("\n📊 Summary:")
        print(f"   Total species extracted: {len(df)}")
        print(f"   With family info: {len(df[df['family'] != ''])}")
        print(f"   With order info: {len(df[df['order'] != ''])}")
        print(f"   With size info: {len(df[df['max_size'] != ''])}")
        print(f"   With IUCN status: {len(df[df['iucn_status'] != ''])}")
        
        print("\n🎯 Next Steps:")
        print("   1. Review: data/final/fish_mapping_auto_generated.csv")
        print("   2. Manually verify and clean the data")
        print("   3. Add missing information (family, order, etc.)")
        print("   4. Merge with your existing fish_mapping.csv")
        
        return df


if __name__ == "__main__":
    extractor = FishDataExtractor()
    df = extractor.run()