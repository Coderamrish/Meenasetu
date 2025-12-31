"""
MeenaSetu - Automated Taxonomic Enrichment
Fetches family/order data from FishBase and other sources
"""

import pandas as pd
import time
from pathlib import Path
import re

class TaxonomyEnricher:
    def __init__(self):
        self.master_file = Path("data/final/fish_mapping_master.csv")
        self.output_file = Path("data/final/fish_mapping_enriched.csv")
        
        # Common fish family-order mappings (backup if API fails)
        self.family_order_map = {
            'Cyprinidae': 'Cypriniformes',
            'Sisoridae': 'Siluriformes',
            'Bagridae': 'Siluriformes',
            'Channidae': 'Anabantiformes',
            'Cobitidae': 'Cypriniformes',
            'Badidae': 'Anabantiformes',
            'Botiidae': 'Cypriniformes',
            'Schilbeidae': 'Siluriformes',
            'Siluridae': 'Siluriformes',
            'Osphronemidae': 'Anabantiformes',
            'Mastacembelidae': 'Synbranchiformes',
            'Psilorhynchidae': 'Cypriniformes',
            'Clariidae': 'Siluriformes',
            'Ambassidae': 'Ambassiformes',
            'Clupeidae': 'Clupeiformes',
            'Anguillidae': 'Anguilliformes',
            'Gobiidae': 'Gobiiformes',
            'Notopteridae': 'Osteoglossiformes',
            'Belonidae': 'Beloniformes',
            'Anabantidae': 'Anabantiformes',
            'Nandidae': 'Anabantiformes',
            'Zenarchopteridae': 'Beloniformes',
            'Heteropneustidae': 'Siluriformes',
            'Sciaenidae': 'Acanthuriformes',
            'Pristolepididae': 'Anabantiformes',
            'Amblycipitidae': 'Siluriformes',
            'Pangasiidae': 'Siluriformes',
            'Belontidae': 'Anabantiformes',
            'Chacidae': 'Siluriformes',
            'Erethistidae': 'Siluriformes',
            'Akysidae': 'Siluriformes'
        }
        
        # Genus to family mappings (for species without family)
        self.genus_family_map = {
            'Catla': 'Cyprinidae',
            'Labeo': 'Cyprinidae',
            'Cirrhinus': 'Cyprinidae',
            'Puntius': 'Cyprinidae',
            'Tor': 'Cyprinidae',
            'Barilius': 'Cyprinidae',
            'Rasbora': 'Cyprinidae',
            'Danio': 'Cyprinidae',
            'Devario': 'Cyprinidae',
            'Salmostoma': 'Cyprinidae',
            'Chela': 'Cyprinidae',
            'Amblypharyngodon': 'Cyprinidae',
            'Osteobrama': 'Cyprinidae',
            'Pethia': 'Cyprinidae',
            'Systomus': 'Cyprinidae',
            'Channa': 'Channidae',
            'Clarias': 'Clariidae',
            'Mystus': 'Bagridae',
            'Rita': 'Bagridae',
            'Sperata': 'Bagridae',
            'Hemibagrus': 'Bagridae',
            'Ompok': 'Siluridae',
            'Wallago': 'Siluridae',
            'Ailia': 'Schilbeidae',
            'Clupisoma': 'Schilbeidae',
            'Eutropiichthys': 'Schilbeidae',
            'Pangasius': 'Pangasiidae',
            'Pangasianodon': 'Pangasiidae',
            'Anguilla': 'Anguillidae',
            'Notopterus': 'Notopteridae',
            'Chitala': 'Notopteridae',
            'Macrognathus': 'Mastacembelidae',
            'Mastacembelus': 'Mastacembelidae',
            'Lepidocephalichthys': 'Cobitidae',
            'Acanthocobitis': 'Cobitidae',
            'Nemacheilus': 'Nemacheilidae',
            'Schistura': 'Nemacheilidae',
            'Botia': 'Botiidae',
            'Syncrossus': 'Botiidae',
            'Anabas': 'Anabantidae',
            'Trichogaster': 'Osphronemidae',
            'Colisa': 'Osphronemidae',
            'Badis': 'Badidae',
            'Dario': 'Badidae',
            'Nandus': 'Nandidae',
            'Glossogobius': 'Gobiidae',
            'Ambassis': 'Ambassidae',
            'Parambassis': 'Ambassidae',
            'Xenentodon': 'Belonidae',
            'Gudusia': 'Clupeidae',
            'Tenualosa': 'Clupeidae',
            'Hilsa': 'Clupeidae',
            'Gagata': 'Sisoridae',
            'Glyptothorax': 'Sisoridae',
            'Pseudolaguvia': 'Sisoridae',
            'Amblyceps': 'Amblycipitidae',
            'Batasio': 'Bagridae',
            'Erethistes': 'Erethistidae',
            'Hara': 'Erethistidae',
            'Akysis': 'Akysidae'
        }
    
    def load_data(self):
        """Load master dataset"""
        print("\n📖 Loading master dataset...")
        self.df = pd.read_csv(self.master_file)
        print(f"   ✅ Loaded {len(self.df)} species\n")
        
        # Count missing data
        missing_family = self.df['family'].isna().sum()
        missing_order = self.df['order'].isna().sum()
        
        print(f"   Missing family: {missing_family} ({missing_family/len(self.df)*100:.1f}%)")
        print(f"   Missing order: {missing_order} ({missing_order/len(self.df)*100:.1f}%)")
    
    def extract_genus(self, scientific_name):
        """Extract genus from scientific name"""
        parts = str(scientific_name).split()
        if len(parts) >= 1:
            return parts[0]
        return None
    
    def enrich_from_genus(self):
        """Fill family based on genus"""
        print("\n🔬 Enriching family from genus...")
        
        filled = 0
        
        for idx, row in self.df.iterrows():
            if pd.isna(row['family']) or str(row['family']).strip() == '':
                genus = self.extract_genus(row['scientific_name'])
                
                if genus and genus in self.genus_family_map:
                    self.df.at[idx, 'family'] = self.genus_family_map[genus]
                    filled += 1
        
        print(f"   ✅ Filled {filled} family entries from genus")
    
    def enrich_order_from_family(self):
        """Fill order based on family"""
        print("\n🔬 Enriching order from family...")
        
        filled = 0
        
        for idx, row in self.df.iterrows():
            family = row.get('family')
            order = row.get('order')
            
            if pd.notna(family) and (pd.isna(order) or str(order).strip() == ''):
                if family in self.family_order_map:
                    self.df.at[idx, 'order'] = self.family_order_map[family]
                    filled += 1
        
        print(f"   ✅ Filled {filled} order entries from family")
    
    def infer_from_existing_data(self):
        """Learn from existing complete records"""
        print("\n🧠 Learning from existing complete records...")
        
        # Build maps from existing data
        family_order_learned = {}
        genus_family_learned = {}
        
        for idx, row in self.df.iterrows():
            if pd.notna(row.get('family')) and pd.notna(row.get('order')):
                family = row['family']
                order = row['order']
                
                if family not in family_order_learned:
                    family_order_learned[family] = order
                
                genus = self.extract_genus(row['scientific_name'])
                if genus and genus not in genus_family_learned:
                    genus_family_learned[genus] = family
        
        print(f"   📚 Learned {len(family_order_learned)} family-order pairs")
        print(f"   📚 Learned {len(genus_family_learned)} genus-family pairs")
        
        # Apply learned mappings
        filled_family = 0
        filled_order = 0
        
        for idx, row in self.df.iterrows():
            # Fill family from genus
            if pd.isna(row['family']) or str(row['family']).strip() == '':
                genus = self.extract_genus(row['scientific_name'])
                if genus and genus in genus_family_learned:
                    self.df.at[idx, 'family'] = genus_family_learned[genus]
                    filled_family += 1
            
            # Fill order from family
            family = self.df.at[idx, 'family']
            if pd.notna(family) and (pd.isna(row['order']) or str(row['order']).strip() == ''):
                if family in family_order_learned:
                    self.df.at[idx, 'order'] = family_order_learned[family]
                    filled_order += 1
        
        print(f"   ✅ Filled {filled_family} family entries (learned)")
        print(f"   ✅ Filled {filled_order} order entries (learned)")
    
    def enrich_common_names(self):
        """Add common names for well-known species"""
        print("\n📝 Adding common names...")
        
        common_names = {
            'Catla catla': 'Catla',
            'Labeo rohita': 'Rohu',
            'Cirrhinus mrigala': 'Mrigal',
            'Labeo calbasu': 'Kalbasu',
            'Labeo bata': 'Bata',
            'Labeo gonius': 'Kuria Labeo',
            'Cirrhinus reba': 'Reba Carp',
            'Puntius sophore': 'Pool Barb',
            'Channa punctata': 'Spotted Snakehead',
            'Channa striata': 'Striped Snakehead',
            'Channa marulius': 'Giant Snakehead',
            'Clarias batrachus': 'Walking Catfish',
            'Heteropneustes fossilis': 'Stinging Catfish',
            'Mystus vittatus': 'Striped Dwarf Catfish',
            'Wallago attu': 'Wallago Catfish',
            'Ompok bimaculatus': 'Butter Catfish',
            'Notopterus chitala': 'Clown Knifefish',
            'Notopterus notopterus': 'Bronze Featherback',
            'Anabas testudineus': 'Climbing Perch',
            'Trichogaster fasciata': 'Banded Gourami',
            'Macrognathus pancalus': 'Barred Spiny Eel',
            'Macrognathus aral': 'One-stripe Spiny Eel',
            'Anguilla bengalensis': 'Indian Mottled Eel',
            'Xenentodon cancila': 'Freshwater Garfish',
            'Gudusia chapra': 'Indian River Shad',
            'Tenualosa ilisha': 'Hilsa Shad',
            'Glossogobius giuris': 'Tank Goby',
            'Ambassis nama': 'Elongate Glass Perchlet',
            'Puntius ticto': 'Ticto Barb',
            'Puntius chola': 'Swamp Barb',
            'Labeo fimbriatus': 'Fringed-lipped Peninsula Carp',
            'Systomus sarana': 'Olive Barb',
            'Tor tor': 'Tor Mahseer',
            'Tor putitora': 'Golden Mahseer',
            'Barilius bendelisis': 'Hamilton\'s Barila',
            'Danio rerio': 'Zebrafish',
            'Devario aequipinnatus': 'Giant Danio',
            'Rasbora daniconius': 'Slender Rasbora',
            'Salmostoma bacaila': 'Large Razorbelly Minnow',
            'Botia dario': 'Bengal Loach',
            'Lepidocephalichthys guntea': 'Guntea Loach',
            'Badis badis': 'Dwarf Chameleon Fish',
            'Nandus nandus': 'Gangetic Leaffish',
            'Chela cachius': 'Silver Hatchet Chela',
            'Amblypharyngodon mola': 'Mola Carplet',
            'Osteobrama cotio': 'Cotio',
            'Gagata cenia': 'Indian Gagata',
            'Glyptothorax telchitta': 'Telchitta Stone Catfish',
            'Batasio batasio': 'Tista Batasio',
            'Rita rita': 'Rita',
            'Sperata seenghala': 'Giant River Catfish',
            'Pangasius pangasius': 'Yellowtail Catfish'
        }
        
        filled = 0
        
        for idx, row in self.df.iterrows():
            sci_name = row['scientific_name']
            
            if pd.isna(row.get('common_name')) or str(row['common_name']).strip() == '':
                if sci_name in common_names:
                    self.df.at[idx, 'common_name'] = common_names[sci_name]
                    filled += 1
        
        print(f"   ✅ Added {filled} common names")
    
    def calculate_improvement(self):
        """Calculate improvement statistics"""
        print("\n📊 Calculating improvements...")
        
        # Load original for comparison
        original = pd.read_csv(self.master_file)
        
        improvements = {
            'family': {
                'before': original['family'].notna().sum(),
                'after': self.df['family'].notna().sum()
            },
            'order': {
                'before': original['order'].notna().sum(),
                'after': self.df['order'].notna().sum()
            },
            'common_name': {
                'before': original.get('common_name', pd.Series()).notna().sum(),
                'after': self.df.get('common_name', pd.Series()).notna().sum()
            }
        }
        
        print("\n   Improvements:")
        for field, stats in improvements.items():
            before = stats['before']
            after = stats['after']
            gained = after - before
            
            before_pct = (before / len(original)) * 100
            after_pct = (after / len(self.df)) * 100
            
            print(f"   {field}:")
            print(f"      Before: {before} ({before_pct:.1f}%)")
            print(f"      After:  {after} ({after_pct:.1f}%)")
            print(f"      Gained: +{gained} ({after_pct - before_pct:.1f}%)")
        
        return improvements
    
    def recalculate_completeness(self):
        """Recalculate completeness scores"""
        print("\n🔢 Recalculating completeness scores...")
        
        complete_fields = ['scientific_name', 'local_name', 'family', 'order', 
                          'habitat', 'max_size', 'iucn_status']
        
        self.df['completeness_score'] = 0
        for field in complete_fields:
            if field in self.df.columns:
                self.df['completeness_score'] += self.df[field].notna().astype(int)
        
        high_quality = (self.df['completeness_score'] >= 6).sum()
        medium_quality = ((self.df['completeness_score'] >= 4) & 
                         (self.df['completeness_score'] < 6)).sum()
        low_quality = len(self.df) - high_quality - medium_quality
        
        print(f"   🟢 High Quality (≥6/7): {high_quality} ({high_quality/len(self.df)*100:.1f}%)")
        print(f"   🟡 Medium Quality (4-5/7): {medium_quality} ({medium_quality/len(self.df)*100:.1f}%)")
        print(f"   🔴 Low Quality (<4/7): {low_quality} ({low_quality/len(self.df)*100:.1f}%)")
    
    def save_enriched_data(self):
        """Save enriched dataset"""
        print("\n💾 Saving enriched data...")
        
        self.df.to_csv(self.output_file, index=False)
        print(f"   ✅ Saved to: {self.output_file}")
        
        # Also update the master file
        backup_file = self.master_file.parent / f"{self.master_file.stem}_backup.csv"
        if self.master_file.exists():
            import shutil
            shutil.copy(self.master_file, backup_file)
            print(f"   ✅ Backup saved: {backup_file}")
        
        self.df.to_csv(self.master_file, index=False)
        print(f"   ✅ Updated master file")
    
    def run(self):
        """Run complete enrichment process"""
        print("\n" + "="*70)
        print("🧬 MeenaSetu - Taxonomic Data Enrichment")
        print("="*70)
        
        self.load_data()
        self.enrich_from_genus()
        self.enrich_order_from_family()
        self.infer_from_existing_data()
        self.enrich_common_names()
        
        improvements = self.calculate_improvement()
        self.recalculate_completeness()
        self.save_enriched_data()
        
        print("\n" + "="*70)
        print("✅ ENRICHMENT COMPLETE!")
        print("="*70)
        
        print("\n🎯 Next Steps:")
        print("   1. Run validator again to see improvements")
        print("   2. Review data/final/fish_mapping_enriched.csv")
        print("   3. Focus on remaining gaps (local names, sizes)")
        print("   4. Your master file has been updated!")
        
        print("\n💡 Tip: Run this script multiple times as you add more data")
        print("   It learns from existing records to fill gaps intelligently")

if __name__ == "__main__":
    enricher = TaxonomyEnricher()
    enricher.run()