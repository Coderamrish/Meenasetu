"""
MeenaSetu - Local Taxonomy Database Builder
Fast, reliable taxonomy extraction from WB Fish Diversity tables
NO API calls - works offline instantly!
"""

import pandas as pd
import re
from pathlib import Path
from collections import defaultdict

class LocalTaxonomyBuilder:
    def __init__(self):
        self.processed_dir = Path("datasets/processed")
        self.master_file = Path("data/final/fish_mapping_master.csv")
        self.taxonomy_db = {}  # species -> {family, order}
        
        # Known family-order mappings (comprehensive)
        self.family_order_map = {
            # Cypriniformes families
            'Cyprinidae': 'Cypriniformes',
            'Balitoridae': 'Cypriniformes',
            'Cobitidae': 'Cypriniformes',
            'Nemacheilidae': 'Cypriniformes',
            'Botiidae': 'Cypriniformes',
            'Psilorhynchidae': 'Cypriniformes',
            
            # Siluriformes families
            'Bagridae': 'Siluriformes',
            'Sisoridae': 'Siluriformes',
            'Siluridae': 'Siluriformes',
            'Schilbeidae': 'Siluriformes',
            'Schilbedidae': 'Siluriformes',
            'Clariidae': 'Siluriformes',
            'Heteropneustidae': 'Siluriformes',
            'Amblycipitidae': 'Siluriformes',
            'Pangasidae': 'Siluriformes',
            'Pangasiidae': 'Siluriformes',
            'Chacidae': 'Siluriformes',
            'Erethistidae': 'Siluriformes',
            'Akysidae': 'Siluriformes',
            'Plotosidae': 'Siluriformes',
            'Loricariidae': 'Siluriformes',
            'Arridae': 'Siluriformes',
            
            # Perciformes & related orders
            'Ambassidae': 'Ambassiformes',
            'Anabantidae': 'Anabantiformes',
            'Badidae': 'Anabantiformes',
            'Channidae': 'Anabantiformes',
            'Osphronemidae': 'Anabantiformes',
            'Nandidae': 'Anabantiformes',
            'Pristolepididae': 'Anabantiformes',
            'Belontidae': 'Anabantiformes',
            'Gobiidae': 'Gobiiformes',
            'Eleotridae': 'Gobiiformes',
            'Sciaenidae': 'Acanthuriformes',
            'Datniodidae': 'Lobotiformes',
            'Chichlidae': 'Cichliformes',
            
            # Other orders
            'Anguillidae': 'Anguilliformes',
            'Ophichthidae': 'Anguilliformes',
            'Belonidae': 'Beloniformes',
            'Zenarchopteridae': 'Beloniformes',
            'Clupeidae': 'Clupeiformes',
            'Engraulididae': 'Clupeiformes',
            'Aplocheilidae': 'Cyprinodontiformes',
            'Poeciliidae': 'Cyprinodontiformes',
            'Mugilidae': 'Mugiliformes',
            'Notopteridae': 'Osteoglossiformes',
            'Mastacembelidae': 'Synbranchiformes',
            'Synbranchidae': 'Synbranchiformes',
            'Chaudhuriidae': 'Synbranchiformes',
            'Syngnathidae': 'Syngnathiformes',
            'Tetraodontidae': 'Tetraodontiformes'
        }
        
        # Genus to family mappings (comprehensive)
        self.genus_family_map = self._build_genus_family_map()
    
    def _build_genus_family_map(self):
        """Build comprehensive genus-to-family mapping"""
        return {
            # Cyprinidae genera
            'Catla': 'Cyprinidae', 'Labeo': 'Cyprinidae', 'Cirrhinus': 'Cyprinidae',
            'Puntius': 'Cyprinidae', 'Pethia': 'Cyprinidae', 'Systomus': 'Cyprinidae',
            'Tor': 'Cyprinidae', 'Barilius': 'Cyprinidae', 'Rasbora': 'Cyprinidae',
            'Danio': 'Cyprinidae', 'Devario': 'Cyprinidae', 'Salmostoma': 'Cyprinidae',
            'Salmophasia': 'Cyprinidae', 'Chela': 'Cyprinidae', 'Amblypharyngodon': 'Cyprinidae',
            'Osteobrama': 'Cyprinidae', 'Cabdio': 'Cyprinidae', 'Carassius': 'Cyprinidae',
            'Chagunius': 'Cyprinidae', 'Crossocheilus': 'Cyprinidae', 'Ctenopharyngodon': 'Cyprinidae',
            'Cyprinion': 'Cyprinidae', 'Cyprinus': 'Cyprinidae', 'Danionella': 'Cyprinidae',
            'Esomus': 'Cyprinidae', 'Garra': 'Cyprinidae', 'Gibelion': 'Cyprinidae',
            'Hypophthalmichthys': 'Cyprinidae', 'Neolissochilus': 'Cyprinidae', 'Raiamas': 'Cyprinidae',
            'Schizothorax': 'Cyprinidae', 'Securicula': 'Cyprinidae', 'Dawkinsia': 'Cyprinidae',
            'Hypselobarbus': 'Cyprinidae', 'Oreichthys': 'Cyprinidae',
            
            # Channidae
            'Channa': 'Channidae',
            
            # Clariidae
            'Clarias': 'Clariidae',
            
            # Bagridae
            'Mystus': 'Bagridae', 'Rita': 'Bagridae', 'Sperata': 'Bagridae',
            'Hemibagrus': 'Bagridae', 'Batasio': 'Bagridae', 'Horabagrus': 'Bagridae',
            
            # Siluridae
            'Ompok': 'Siluridae', 'Wallago': 'Siluridae', 'Belodontichthys': 'Siluridae',
            
            # Schilbeidae
            'Ailia': 'Schilbeidae', 'Clupisoma': 'Schilbeidae', 'Eutropiichthys': 'Schilbeidae',
            'Neotropius': 'Schilbeidae', 'Pseudeutropius': 'Schilbeidae', 'Proeutropiichthys': 'Schilbeidae',
            
            # Sisoridae
            'Gagata': 'Sisoridae', 'Glyptothorax': 'Sisoridae', 'Pseudolaguvia': 'Sisoridae',
            'Sisor': 'Sisoridae', 'Bagarius': 'Sisoridae', 'Nangra': 'Sisoridae',
            'Exostoma': 'Sisoridae', 'Erethistes': 'Erethistidae', 'Hara': 'Erethistidae',
            
            # Pangasiidae
            'Pangasius': 'Pangasiidae', 'Pangasianodon': 'Pangasiidae',
            
            # Heteropneustidae
            'Heteropneustes': 'Heteropneustidae',
            
            # Cobitidae & Nemacheilidae
            'Lepidocephalichthys': 'Cobitidae', 'Acanthocobitis': 'Cobitidae',
            'Nemacheilus': 'Nemacheilidae', 'Schistura': 'Nemacheilidae',
            
            # Botiidae
            'Botia': 'Botiidae', 'Syncrossus': 'Botiidae',
            
            # Anabantidae
            'Anabas': 'Anabantidae',
            
            # Osphronemidae
            'Trichogaster': 'Osphronemidae', 'Colisa': 'Osphronemidae',
            'Trichopodus': 'Osphronemidae',
            
            # Badidae
            'Badis': 'Badidae', 'Dario': 'Badidae',
            
            # Nandidae
            'Nandus': 'Nandidae', 'Polycentropsis': 'Nandidae',
            
            # Gobiidae
            'Glossogobius': 'Gobiidae', 'Sicyopterus': 'Gobiidae',
            
            # Ambassidae
            'Ambassis': 'Ambassidae', 'Parambassis': 'Ambassidae', 'Chanda': 'Ambassidae',
            
            # Belonidae
            'Xenentodon': 'Belonidae', 'Strongylura': 'Belonidae',
            
            # Clupeidae
            'Gudusia': 'Clupeidae', 'Tenualosa': 'Clupeidae', 'Hilsa': 'Clupeidae',
            'Corica': 'Clupeidae', 'Gonialosa': 'Clupeidae',
            
            # Engraulididae
            'Setipinna': 'Engraulididae', 'Stolephorus': 'Engraulididae',
            
            # Anguillidae
            'Anguilla': 'Anguillidae',
            
            # Notopteridae
            'Notopterus': 'Notopteridae', 'Chitala': 'Notopteridae',
            
            # Mastacembelidae
            'Macrognathus': 'Mastacembelidae', 'Mastacembelus': 'Mastacembelidae',
            
            # Psilorhynchidae
            'Psilorhynchus': 'Psilorhynchidae',
            
            # Others
            'Amblyceps': 'Amblycipitidae', 'Akysis': 'Akysidae',
            'Chaca': 'Chacidae', 'Monopterus': 'Synbranchidae',
            'Pisodonophis': 'Ophichthidae', 'Aplocheilus': 'Aplocheilidae',
            'Oryzias': 'Adrianichthyidae', 'Rhinomugil': 'Mugilidae',
        }
    
    def parse_wb_tables_hierarchical(self):
        """Parse WB tables with proper hierarchy tracking"""
        print("🔍 Parsing WB Fish Diversity tables (hierarchical)...\n")
        
        table_files = [
            'FreshwaterfishdiversityofWestBengal_table_1.csv',
            'FreshwaterfishdiversityofWestBengal_table_2.csv',
            'FreshwaterfishdiversityofWestBengal_table_3.csv',
            'FreshwaterfishdiversityofWestBengal_table_4.csv',
            'FreshwaterfishdiversityofWestBengal_table_5.csv',
            'FreshwaterfishdiversityofWestBengal_table_6.csv',
        ]
        
        current_order = None
        current_family = None
        total_extracted = 0
        
        for filename in table_files:
            filepath = self.processed_dir / filename
            if not filepath.exists():
                continue
            
            print(f"📄 {filename}")
            df = pd.read_csv(filepath)
            
            # Get first column (contains the hierarchy)
            col = df.columns[0]
            
            for _, row in df.iterrows():
                cell = str(row[col]).strip()
                
                if cell == 'nan' or not cell:
                    continue
                
                # Check for Order
                if 'Order:' in cell or cell.startswith('Order'):
                    match = re.search(r'Order[:\s]+([A-Za-z]+)', cell, re.IGNORECASE)
                    if match:
                        current_order = match.group(1).strip()
                    continue
                
                # Check for Family
                if 'Family:' in cell or cell.startswith('Family'):
                    match = re.search(r'Family[:\s]+([A-Za-z]+)', cell, re.IGNORECASE)
                    if match:
                        current_family = match.group(1).strip()
                    continue
                
                # Extract species
                species_match = re.match(r'^([A-Z][a-z]+ [a-z]+)', cell)
                if species_match:
                    species = species_match.group(1).strip()
                    
                    self.taxonomy_db[species] = {
                        'family': current_family,
                        'order': current_order
                    }
                    total_extracted += 1
            
            print(f"   ✅ Extracted species with taxonomy\n")
        
        print(f"📊 Total species in taxonomy DB: {total_extracted}\n")
        return total_extracted
    
    def enrich_from_genus(self):
        """Add family based on genus for species not in tables"""
        print("🧬 Enriching from genus mappings...")
        
        added = 0
        
        for species in self.taxonomy_db.keys():
            if not self.taxonomy_db[species]['family']:
                genus = species.split()[0]
                if genus in self.genus_family_map:
                    self.taxonomy_db[species]['family'] = self.genus_family_map[genus]
                    added += 1
        
        print(f"   ✅ Added {added} families from genus\n")
    
    def add_orders_from_families(self):
        """Add orders based on family"""
        print("📚 Adding orders from families...")
        
        added = 0
        
        for species, data in self.taxonomy_db.items():
            family = data.get('family')
            if family and not data.get('order'):
                if family in self.family_order_map:
                    self.taxonomy_db[species]['order'] = self.family_order_map[family]
                    added += 1
        
        print(f"   ✅ Added {added} orders\n")
    
    def apply_to_master(self):
        """Apply taxonomy to master dataset"""
        print("🔄 Applying taxonomy to master dataset...\n")
        
        df = pd.read_csv(self.master_file)
        
        # Clean invalid species first
        before = len(df)
        df = df[df['scientific_name'].apply(self._is_valid_species)]
        df = df.reset_index(drop=True)
        removed = before - len(df)
        
        if removed > 0:
            print(f"   🧹 Removed {removed} invalid entries")
        
        updated_family = 0
        updated_order = 0
        
        for idx, row in df.iterrows():
            sci_name = self._clean_species_name(row['scientific_name'])
            
            # Try exact match first
            taxonomy = self.taxonomy_db.get(sci_name)
            
            # If not found, try genus-based lookup
            if not taxonomy:
                genus = sci_name.split()[0]
                if genus in self.genus_family_map:
                    family = self.genus_family_map[genus]
                    order = self.family_order_map.get(family)
                    taxonomy = {'family': family, 'order': order}
            
            if taxonomy:
                # Update family
                if (pd.isna(row['family']) or not str(row['family']).strip()) and taxonomy.get('family'):
                    df.at[idx, 'family'] = taxonomy['family']
                    updated_family += 1
                
                # Update order
                if (pd.isna(row['order']) or not str(row['order']).strip()) and taxonomy.get('order'):
                    df.at[idx, 'order'] = taxonomy['order']
                    updated_order += 1
        
        print(f"   ✅ Updated {updated_family} family entries")
        print(f"   ✅ Updated {updated_order} order entries\n")
        
        return df, updated_family, updated_order
    
    def _is_valid_species(self, name):
        """Check if valid species name"""
        name = str(name).strip()
        pattern = r'^[A-Z][a-z]+ [a-z]+'
        return bool(re.match(pattern, name))
    
    def _clean_species_name(self, name):
        """Extract clean species name"""
        match = re.match(r'^([A-Z][a-z]+ [a-z]+)', str(name))
        return match.group(1) if match else name
    
    def show_statistics(self, df):
        """Show coverage statistics"""
        print("📊 Taxonomy Coverage:\n")
        
        total = len(df)
        has_family = df['family'].notna().sum()
        has_order = df['order'].notna().sum()
        
        print(f"   Total species: {total}")
        print(f"   Family: {has_family} ({has_family/total*100:.1f}%)")
        print(f"   Order: {has_order} ({has_order/total*100:.1f}%)")
        
        # Quality scores
        complete_fields = ['scientific_name', 'family', 'order', 'habitat', 'iucn_status']
        df['temp_score'] = sum(df[f].notna().astype(int) for f in complete_fields if f in df.columns)
        
        high = (df['temp_score'] >= 4).sum()
        print(f"\n   🟢 High quality (≥4/5 core fields): {high} ({high/total*100:.1f}%)")
    
    def save_results(self, df):
        """Save enriched data"""
        print("\n💾 Saving results...")
        
        # Backup
        backup = self.master_file.parent / f"{self.master_file.stem}_pre_local_taxonomy.csv"
        if self.master_file.exists():
            import shutil
            shutil.copy(self.master_file, backup)
            print(f"   ✅ Backup: {backup}")
        
        # Save enriched
        output = self.master_file.parent / "fish_mapping_local_enriched.csv"
        df.to_csv(output, index=False)
        print(f"   ✅ Enriched: {output}")
        
        # Update master
        df.to_csv(self.master_file, index=False)
        print(f"   ✅ Updated master")
        
        # Save taxonomy DB
        tax_df = pd.DataFrame([
            {'species': k, 'family': v['family'], 'order': v['order']}
            for k, v in self.taxonomy_db.items()
        ])
        tax_file = self.master_file.parent / "local_taxonomy_database.csv"
        tax_df.to_csv(tax_file, index=False)
        print(f"   ✅ Taxonomy DB: {tax_file}")
    
    def run(self):
        """Run complete local taxonomy build"""
        print("\n" + "="*70)
        print("🧬 MeenaSetu - Local Taxonomy Builder")
        print("="*70)
        print("Fast, reliable, offline taxonomy enrichment")
        print("="*70 + "\n")
        
        # Build taxonomy database
        self.parse_wb_tables_hierarchical()
        self.enrich_from_genus()
        self.add_orders_from_families()
        
        # Apply to master
        df, fam_count, ord_count = self.apply_to_master()
        
        # Show results
        self.show_statistics(df)
        self.save_results(df)
        
        print("\n" + "="*70)
        print("✅ LOCAL TAXONOMY BUILD COMPLETE!")
        print("="*70)
        
        print(f"\n📈 Summary:")
        print(f"   • Built taxonomy database for {len(self.taxonomy_db)} species")
        print(f"   • Updated {fam_count} family entries")
        print(f"   • Updated {ord_count} order entries")
        print(f"   • 100% offline - no API calls!")
        
        print("\n🎯 Next Steps:")
        print("   1. Run validator to see improvements")
        print("   2. Review: data/final/fish_mapping_local_enriched.csv")
        print("   3. Check: data/final/local_taxonomy_database.csv")

if __name__ == "__main__":
    builder = LocalTaxonomyBuilder()
    builder.run()