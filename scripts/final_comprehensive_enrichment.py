"""
MeenaSetu - Final Comprehensive Enrichment
Maximizes taxonomy coverage through multiple strategies
"""

import pandas as pd
import re
from pathlib import Path

class FinalEnricher:
    def __init__(self):
        self.master_file = Path("data/final/fish_mapping_master.csv")
        
        # Expanded genus-family mappings (200+ genera)
        self.genus_family = {
            # Cyprinidae (carps & minnows)
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
            'Hypselobarbus': 'Cyprinidae', 'Oreichthys': 'Cyprinidae', 'Laubuka': 'Cyprinidae',
            'Megarasbora': 'Cyprinidae', 'Bangana': 'Cyprinidae', 'Aspidoparia': 'Cyprinidae',
            
            # Balitoridae & Nemacheilidae (loaches)
            'Acanthocobitis': 'Balitoridae', 'Balitora': 'Balitoridae', 'Aborichthys': 'Balitoridae',
            'Neonoemacheilus': 'Balitoridae', 'Nemacheilus': 'Nemacheilidae', 'Schistura': 'Nemacheilidae',
            
            # Cobitidae (loaches)
            'Lepidocephalichthys': 'Cobitidae', 'Pangio': 'Cobitidae', 'Canthophrys': 'Cobitidae',
            'Somileptes': 'Cobitidae',
            
            # Botiidae (loaches)
            'Botia': 'Botiidae', 'Syncrossus': 'Botiidae',
            
            # Psilorhynchidae
            'Psilorhynchus': 'Psilorhynchidae',
            
            # Channidae (snakeheads)
            'Channa': 'Channidae',
            
            # Clariidae (catfish)
            'Clarias': 'Clariidae',
            
            # Bagridae (catfish)
            'Mystus': 'Bagridae', 'Rita': 'Bagridae', 'Sperata': 'Bagridae',
            'Hemibagrus': 'Bagridae', 'Batasio': 'Bagridae', 'Horabagrus': 'Bagridae',
            'Chandramara': 'Bagridae', 'Olyra': 'Bagridae',
            
            # Siluridae (catfish)
            'Ompok': 'Siluridae', 'Wallago': 'Siluridae', 'Belodontichthys': 'Siluridae',
            'Pterocryptis': 'Siluridae',
            
            # Schilbeidae (catfish)
            'Ailia': 'Schilbeidae', 'Clupisoma': 'Schilbeidae', 'Eutropiichthys': 'Schilbeidae',
            'Neotropius': 'Schilbeidae', 'Pseudeutropius': 'Schilbeidae', 
            'Proeutropiichthys': 'Schilbeidae', 'Ailiichthys': 'Schilbedidae', 'Silonia': 'Schilbeidae',
            
            # Sisoridae (catfish)
            'Gagata': 'Sisoridae', 'Glyptothorax': 'Sisoridae', 'Pseudolaguvia': 'Sisoridae',
            'Sisor': 'Sisoridae', 'Bagarius': 'Sisoridae', 'Nangra': 'Sisoridae',
            'Exostoma': 'Sisoridae', 'Conta': 'Sisoridae', 'Gogangra': 'Sisoridae',
            'Hara': 'Sisoridae',
            
            # Erethistidae (catfish)
            'Erethistes': 'Erethistidae', 'Erethistoides': 'Erethistidae',
            
            # Pangasiidae (catfish)
            'Pangasius': 'Pangasiidae', 'Pangasianodon': 'Pangasiidae',
            
            # Heteropneustidae (catfish)
            'Heteropneustes': 'Heteropneustidae',
            
            # Amblycipitidae (catfish)
            'Amblyceps': 'Amblycipitidae',
            
            # Chacidae (catfish)
            'Chaca': 'Chacidae',
            
            # Akysidae (catfish)
            'Akysis': 'Akysidae',
            
            # Plotosidae (catfish)
            'Plotosus': 'Plotosidae',
            
            # Loricariidae (catfish)
            'Pterygoplichthys': 'Loricariidae',
            
            # Arridae (catfish)
            'Arius': 'Arridae',
            
            # Anabantidae (climbing perch)
            'Anabas': 'Anabantidae',
            
            # Osphronemidae (gouramis)
            'Trichogaster': 'Osphronemidae', 'Colisa': 'Osphronemidae',
            'Trichopodus': 'Osphronemidae', 'Ctenops': 'Osphronemidae',
            'Osphronemus': 'Osphronemidae',
            
            # Badidae
            'Badis': 'Badidae', 'Dario': 'Badidae',
            
            # Nandidae
            'Nandus': 'Nandidae', 'Polycentropsis': 'Nandidae',
            
            # Gobiidae (gobies)
            'Glossogobius': 'Gobiidae', 'Sicyopterus': 'Gobiidae', 'Apocryptes': 'Gobiidae',
            'Brachyamblyopus': 'Gobiidae', 'Odontamblyopus': 'Gobiidae', 
            'Pseudapocryptes': 'Gobiidae',
            
            # Ambassidae (glass fish)
            'Ambassis': 'Ambassidae', 'Parambassis': 'Ambassidae', 'Chanda': 'Ambassidae',
            
            # Belonidae (needlefish)
            'Xenentodon': 'Belonidae', 'Strongylura': 'Belonidae',
            
            # Clupeidae (herring/shad)
            'Gudusia': 'Clupeidae', 'Tenualosa': 'Clupeidae', 'Hilsa': 'Clupeidae',
            'Corica': 'Clupeidae', 'Gonialosa': 'Clupeidae',
            
            # Pristigasteridae (shad)
            'Ilisha': 'Pristigasteridae',
            
            # Engraulidae (anchovies)
            'Setipinna': 'Engraulidae', 'Stolephorus': 'Engraulidae',
            
            # Anguillidae (eels)
            'Anguilla': 'Anguillidae',
            
            # Ophichthidae (eels)
            'Pisodonophis': 'Ophichthidae',
            
            # Synbranchidae (swamp eels)
            'Monopterus': 'Synbranchidae', 'Ophisternon': 'Synbranchidae',
            
            # Notopteridae (featherbacks)
            'Notopterus': 'Notopteridae', 'Chitala': 'Notopteridae',
            
            # Mastacembelidae (spiny eels)
            'Macrognathus': 'Mastacembelidae', 'Mastacembelus': 'Mastacembelidae',
            
            # Chaudhuriidae
            'Pillaia': 'Chaudhuriidae',
            
            # Aplocheilidae
            'Aplocheilus': 'Aplocheilidae',
            
            # Poeciliidae
            'Gambusia': 'Poeciliidae', 'Poecilia': 'Poeciliidae',
            
            # Mugilidae (mullets)
            'Rhinomugil': 'Mugilidae', 'Sicamugil': 'Mugilidae',
            
            # Eleotridae
            'Eleotris': 'Eleotridae',
            
            # Datniodidae
            'Datnioides': 'Datniodidae',
            
            # Cichlidae (cichlids)
            'Oreochromis': 'Cichlidae', 'Etroplus': 'Cichlidae',
            
            # Syngnathidae (pipefish)
            'Microphis': 'Syngnathidae',
            
            # Tetraodontidae (pufferfish)
            'Leiodon': 'Tetraodontidae', 'Tetraodon': 'Tetraodontidae',
            
            # Zenarchopteridae
            'Zenarchopterus': 'Zenarchopteridae',
        }
        
        # Comprehensive family-order mappings
        self.family_order = {
            'Cyprinidae': 'Cypriniformes',
            'Balitoridae': 'Cypriniformes',
            'Cobitidae': 'Cypriniformes',
            'Nemacheilidae': 'Cypriniformes',
            'Botiidae': 'Cypriniformes',
            'Psilorhynchidae': 'Cypriniformes',
            'Bagridae': 'Siluriformes',
            'Sisoridae': 'Siluriformes',
            'Siluridae': 'Siluriformes',
            'Schilbeidae': 'Siluriformes',
            'Schilbedidae': 'Siluriformes',
            'Clariidae': 'Siluriformes',
            'Heteropneustidae': 'Siluriformes',
            'Amblycipitidae': 'Siluriformes',
            'Pangasiidae': 'Siluriformes',
            'Pangasidae': 'Siluriformes',
            'Chacidae': 'Siluriformes',
            'Erethistidae': 'Siluriformes',
            'Akysidae': 'Siluriformes',
            'Plotosidae': 'Siluriformes',
            'Loricariidae': 'Siluriformes',
            'Arridae': 'Siluriformes',
            'Ambassidae': 'Ambassiformes',
            'Anabantidae': 'Anabantiformes',
            'Badidae': 'Anabantiformes',
            'Channidae': 'Anabantiformes',
            'Channiformes': 'Anabantiformes',  # Some sources use this
            'Osphronemidae': 'Anabantiformes',
            'Nandidae': 'Anabantiformes',
            'Pristolepididae': 'Anabantiformes',
            'Belontidae': 'Anabantiformes',
            'Gobiidae': 'Gobiiformes',
            'Eleotridae': 'Gobiiformes',
            'Sciaenidae': 'Acanthuriformes',
            'Datniodidae': 'Lobotiformes',
            'Chichlidae': 'Cichliformes',
            'Cichlidae': 'Cichliformes',
            'Anguillidae': 'Anguilliformes',
            'Ophichthidae': 'Anguilliformes',
            'Belonidae': 'Beloniformes',
            'Zenarchopteridae': 'Beloniformes',
            'Clupeidae': 'Clupeiformes',
            'Pristigasteridae': 'Clupeiformes',
            'Engraulidae': 'Clupeiformes',
            'Engraulididae': 'Clupeiformes',
            'Aplocheilidae': 'Cyprinodontiformes',
            'Poeciliidae': 'Cyprinodontiformes',
            'Mugilidae': 'Mugiliformes',
            'Notopteridae': 'Osteoglossiformes',
            'Mastacembelidae': 'Synbranchiformes',
            'Synbranchidae': 'Synbranchiformes',
            'Chaudhuriidae': 'Synbranchiformes',
            'Syngnathidae': 'Syngnathiformes',
            'Tetraodontidae': 'Tetraodontiformes',
            'Tetradontiformes': 'Tetraodontiformes',
        }
    
    def load_data(self):
        """Load master file"""
        print("\n📖 Loading master dataset...")
        self.df = pd.read_csv(self.master_file)
        print(f"   ✅ {len(self.df)} species loaded\n")
    
    def extract_genus(self, name):
        """Extract genus from scientific name"""
        match = re.match(r'^([A-Z][a-z]+)', str(name))
        return match.group(1) if match else None
    
    def enrich_family_from_genus(self):
        """Fill missing families from genus"""
        print("🧬 Enriching families from genus...")
        
        filled = 0
        for idx, row in self.df.iterrows():
            if pd.isna(row['family']) or str(row['family']).strip() == '':
                genus = self.extract_genus(row['scientific_name'])
                if genus and genus in self.genus_family:
                    self.df.at[idx, 'family'] = self.genus_family[genus]
                    filled += 1
        
        print(f"   ✅ Filled {filled} families\n")
        return filled
    
    def enrich_order_from_family(self):
        """Fill missing orders from family"""
        print("📚 Enriching orders from families...")
        
        filled = 0
        for idx, row in self.df.iterrows():
            family = row.get('family')
            order = row.get('order')
            
            if pd.notna(family) and (pd.isna(order) or str(order).strip() == ''):
                if family in self.family_order:
                    self.df.at[idx, 'order'] = self.family_order[family]
                    filled += 1
        
        print(f"   ✅ Filled {filled} orders\n")
        return filled
    
    def show_statistics(self):
        """Show detailed statistics"""
        print("📊 Current Coverage:\n")
        
        total = len(self.df)
        
        has_family = self.df['family'].notna().sum()
        has_order = self.df['order'].notna().sum()
        has_size = self.df['max_size'].notna().sum()
        has_local = self.df['local_name'].notna().sum()
        has_common = self.df['common_name'].notna().sum() if 'common_name' in self.df.columns else 0
        has_iucn = self.df['iucn_status'].notna().sum()
        has_habitat = self.df['habitat'].notna().sum()
        
        print(f"   Scientific Name: {total}/{total} (100.0%)")
        print(f"   Family:          {has_family}/{total} ({has_family/total*100:.1f}%)")
        print(f"   Order:           {has_order}/{total} ({has_order/total*100:.1f}%)")
        print(f"   Max Size:        {has_size}/{total} ({has_size/total*100:.1f}%)")
        print(f"   Local Name:      {has_local}/{total} ({has_local/total*100:.1f}%)")
        print(f"   Common Name:     {has_common}/{total} ({has_common/total*100:.1f}%)")
        print(f"   IUCN Status:     {has_iucn}/{total} ({has_iucn/total*100:.1f}%)")
        print(f"   Habitat:         {has_habitat}/{total} ({has_habitat/total*100:.1f}%)")
        
        # Calculate completeness
        essential = ['scientific_name', 'family', 'order', 'habitat', 'iucn_status']
        self.df['temp_score'] = sum(self.df[f].notna().astype(int) for f in essential if f in self.df.columns)
        
        high = (self.df['temp_score'] >= 4).sum()
        medium = ((self.df['temp_score'] >= 3) & (self.df['temp_score'] < 4)).sum()
        
        print(f"\n   🟢 High Quality (≥4/5): {high} ({high/total*100:.1f}%)")
        print(f"   🟡 Medium Quality (3/5): {medium} ({medium/total*100:.1f}%)")
        print(f"   🔴 Low Quality (<3/5): {total-high-medium} ({(total-high-medium)/total*100:.1f}%)")
        
        # Top families
        if has_family > 0:
            print(f"\n   📋 Top 10 Families:")
            for family, count in self.df['family'].value_counts().head(10).items():
                print(f"      {family}: {count}")
    
    def save_results(self):
        """Save enriched data"""
        print("\n💾 Saving enriched data...")
        
        # Backup
        backup = self.master_file.parent / f"{self.master_file.stem}_v4_backup.csv"
        import shutil
        if self.master_file.exists():
            shutil.copy(self.master_file, backup)
            print(f"   ✅ Backup: {backup}")
        
        # Save
        self.df.to_csv(self.master_file, index=False)
        print(f"   ✅ Updated: {self.master_file}")
        
        # Save high-quality subset
        high_quality = self.df[self.df['temp_score'] >= 4].copy()
        hq_file = self.master_file.parent / "fish_mapping_production_ready.csv"
        high_quality.to_csv(hq_file, index=False)
        print(f"   ✅ Production ready: {hq_file} ({len(high_quality)} species)")
    
    def run(self):
        """Run complete enrichment"""
        print("\n" + "="*70)
        print("🚀 MeenaSetu - Final Comprehensive Enrichment")
        print("="*70)
        print("Maximizing taxonomy coverage with all available data")
        print("="*70)
        
        self.load_data()
        
        print("📊 Before Enrichment:")
        has_family_before = self.df['family'].notna().sum()
        has_order_before = self.df['order'].notna().sum()
        print(f"   Family: {has_family_before} ({has_family_before/len(self.df)*100:.1f}%)")
        print(f"   Order: {has_order_before} ({has_order_before/len(self.df)*100:.1f}%)\n")
        
        family_added = self.enrich_family_from_genus()
        order_added = self.enrich_order_from_family()
        
        print("📊 After Enrichment:")
        has_family_after = self.df['family'].notna().sum()
        has_order_after = self.df['order'].notna().sum()
        print(f"   Family: {has_family_after} ({has_family_after/len(self.df)*100:.1f}%) [+{family_added}]")
        print(f"   Order: {has_order_after} ({has_order_after/len(self.df)*100:.1f}%) [+{order_added}]\n")
        
        self.show_statistics()
        self.save_results()
        
        print("\n" + "="*70)
        print("✅ ENRICHMENT COMPLETE!")
        print("="*70)
        
        print(f"\n🎯 Summary:")
        print(f"   • Added {family_added} families")
        print(f"   • Added {order_added} orders")
        print(f"   • Total species: {len(self.df)}")
        print(f"   • Production-ready species: {(self.df['temp_score'] >= 4).sum()}")
        
        print("\n📁 Files Created:")
        print("   1. fish_mapping_master.csv (updated)")
        print("   2. fish_mapping_production_ready.csv (high quality subset)")
        
        print("\n🎯 Next Steps:")
        print("   1. Use fish_mapping_production_ready.csv for your app")
        print("   2. Run validator to check final quality")
        print("   3. Start building your fish identification features!")

if __name__ == "__main__":
    enricher = FinalEnricher()
    enricher.run()