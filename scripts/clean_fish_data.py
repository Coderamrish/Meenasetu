"""
MeenaSetu - Data Cleaner & Enricher
Cleans the 1420 species and fills missing family/order info
"""

import pandas as pd
import re
from collections import Counter

class FishDataCleaner:
    def __init__(self, input_file="data/final/fish_mapping_auto_generated.csv"):
        self.df = pd.read_csv(input_file)
        self.original_count = len(self.df)
        
        # Comprehensive genus to family mapping
        self.genus_family_map = {
            # Cyprinidae (Carps and Minnows)
            'Labeo': 'Cyprinidae', 'Catla': 'Cyprinidae', 'Cirrhinus': 'Cyprinidae',
            'Puntius': 'Cyprinidae', 'Pethia': 'Cyprinidae', 'Barilius': 'Cyprinidae',
            'Salmostoma': 'Cyprinidae', 'Chela': 'Cyprinidae', 'Amblypharyngodon': 'Cyprinidae',
            'Osteobrama': 'Cyprinidae', 'Rasbora': 'Cyprinidae', 'Danio': 'Cyprinidae',
            'Devario': 'Cyprinidae', 'Esomus': 'Cyprinidae', 'Garra': 'Cyprinidae',
            'Tor': 'Cyprinidae', 'Neolissochilus': 'Cyprinidae', 'Systomus': 'Cyprinidae',
            'Cabdio': 'Cyprinidae', 'Cyprinus': 'Cyprinidae', 'Ctenopharyngodon': 'Cyprinidae',
            'Hypophthalmichthys': 'Cyprinidae', 'Aspidoparia': 'Cyprinidae',
            
            # Bagridae (Bagrid Catfishes)
            'Mystus': 'Bagridae', 'Rita': 'Bagridae', 'Sperata': 'Bagridae',
            'Hemibagrus': 'Bagridae', 'Horabagrus': 'Bagridae', 'Batasio': 'Bagridae',
            
            # Siluridae (Sheatfishes)
            'Wallago': 'Siluridae', 'Ompok': 'Siluridae', 'Silurus': 'Siluridae',
            'Pterocryptis': 'Siluridae',
            
            # Schilbeidae (Schilbid Catfishes)
            'Ailia': 'Schilbeidae', 'Clupisoma': 'Schilbeidae', 'Eutropiichthys': 'Schilbeidae',
            'Proeutropiichthys': 'Schilbeidae', 'Silonia': 'Schilbeidae',
            
            # Channidae (Snakeheads)
            'Channa': 'Channidae',
            
            # Anabantidae (Climbing Perches)
            'Anabas': 'Anabantidae',
            
            # Osphronemidae (Gouramies)
            'Trichogaster': 'Osphronemidae', 'Trichopodus': 'Osphronemidae',
            'Colisa': 'Osphronemidae', 'Ctenops': 'Osphronemidae',
            
            # Badidae (Badids)
            'Badis': 'Badidae', 'Dario': 'Badidae',
            
            # Ambassidae (Asiatic Glassfishes)
            'Chanda': 'Ambassidae', 'Parambassis': 'Ambassidae', 'Pseudambassis': 'Ambassidae',
            
            # Gobiidae (Gobies)
            'Glossogobius': 'Gobiidae', 'Awaous': 'Gobiidae', 'Rhinogobius': 'Gobiidae',
            
            # Mastacembelidae (Spiny Eels)
            'Macrognathus': 'Mastacembelidae', 'Mastacembelus': 'Mastacembelidae',
            
            # Heteropneustidae (Airsac Catfishes)
            'Heteropneustes': 'Heteropneustidae',
            
            # Clariidae (Airbreathing Catfishes)
            'Clarias': 'Clariidae',
            
            # Notopteridae (Featherbacks)
            'Notopterus': 'Notopteridae', 'Chitala': 'Notopteridae',
            
            # Cobitidae (Loaches)
            'Lepidocephalichthys': 'Cobitidae', 'Acanthocobitis': 'Cobitidae',
            'Nemacheilus': 'Cobitidae', 'Pangio': 'Cobitidae',
            
            # Botiidae (Loaches)
            'Botia': 'Botiidae', 'Syncrossus': 'Botiidae',
            
            # Balitoridae (River Loaches)
            'Balitora': 'Balitoridae', 'Bhavania': 'Balitoridae',
            
            # Sisoridae (Sisorid Catfishes)
            'Bagarius': 'Sisoridae', 'Gagata': 'Sisoridae', 'Glyptothorax': 'Sisoridae',
            'Nangra': 'Sisoridae', 'Sisor': 'Sisoridae', 'Erethistes': 'Sisoridae',
            'Pseudolaguvia': 'Sisoridae', 'Hara': 'Sisoridae',
            
            # Psilorhynchidae
            'Psilorhynchus': 'Psilorhynchidae',
            
            # Belonidae (Needlefishes)
            'Xenentodon': 'Belonidae',
            
            # Anguillidae (Freshwater Eels)
            'Anguilla': 'Anguillidae',
            
            # Pristigasteridae
            'Gudusia': 'Pristigasteridae',
            
            # Clupeidae (Herrings)
            'Tenualosa': 'Clupeidae', 'Hilsa': 'Clupeidae',
            
            # Engraulidae (Anchovies)
            'Setipinna': 'Engraulidae', 'Coilia': 'Engraulidae',
            
            # Nandidae
            'Nandus': 'Nandidae',
            
            # Cichlidae
            'Oreochromis': 'Cichlidae', 'Etroplus': 'Cichlidae',
            
            # Synbranchidae
            'Monopterus': 'Synbranchidae', 'Ophisternon': 'Synbranchidae',
            
            # Aplocheilidae
            'Aplocheilus': 'Aplocheilidae',
        }
        
        # Family to Order mapping
        self.family_order_map = {
            'Cyprinidae': 'Cypriniformes',
            'Cobitidae': 'Cypriniformes',
            'Botiidae': 'Cypriniformes',
            'Balitoridae': 'Cypriniformes',
            'Psilorhynchidae': 'Cypriniformes',
            'Bagridae': 'Siluriformes',
            'Siluridae': 'Siluriformes',
            'Schilbeidae': 'Siluriformes',
            'Heteropneustidae': 'Siluriformes',
            'Clariidae': 'Siluriformes',
            'Sisoridae': 'Siluriformes',
            'Channidae': 'Channiformes',
            'Anabantidae': 'Anabantiformes',
            'Osphronemidae': 'Anabantiformes',
            'Badidae': 'Perciformes',
            'Ambassidae': 'Perciformes',
            'Gobiidae': 'Gobiiformes',
            'Mastacembelidae': 'Synbranchiformes',
            'Synbranchidae': 'Synbranchiformes',
            'Notopteridae': 'Osteoglossiformes',
            'Belonidae': 'Beloniformes',
            'Anguillidae': 'Anguilliformes',
            'Pristigasteridae': 'Clupeiformes',
            'Clupeidae': 'Clupeiformes',
            'Engraulidae': 'Clupeiformes',
            'Nandidae': 'Perciformes',
            'Cichlidae': 'Cichliformes',
            'Aplocheilidae': 'Cyprinodontiformes',
        }
    
    def clean_scientific_names(self):
        """Remove invalid entries and clean scientific names"""
        print("\n🧹 Step 1: Cleaning scientific names...")
        
        # Remove rows where scientific_name contains 'Family' or 'Order'
        before = len(self.df)
        self.df = self.df[~self.df['scientific_name'].str.contains('Family|Order|Class|Phylum', case=False, na=False)]
        removed = before - len(self.df)
        print(f"   Removed {removed} invalid entries (Family/Order headers)")
        
        # Remove very short names (less than 5 chars)
        before = len(self.df)
        self.df = self.df[self.df['scientific_name'].str.len() > 5]
        removed = before - len(self.df)
        print(f"   Removed {removed} invalid short names")
        
        # Clean whitespace
        self.df['scientific_name'] = self.df['scientific_name'].str.strip()
        
        print(f"   ✅ Cleaned. Remaining: {len(self.df)} species")
    
    def remove_true_duplicates(self):
        """Remove exact duplicate species"""
        print("\n🔄 Step 2: Removing duplicates...")
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=['scientific_name'], keep='first')
        removed = before - len(self.df)
        print(f"   Removed {removed} duplicate species")
        print(f"   ✅ Unique species: {len(self.df)}")
    
    def fill_family_from_genus(self):
        """Fill missing family information based on genus"""
        print("\n🔬 Step 3: Filling family information...")
        
        filled = 0
        for idx, row in self.df.iterrows():
            if pd.isna(row['family']) or row['family'] == '':
                genus = row['scientific_name'].split()[0]
                if genus in self.genus_family_map:
                    self.df.at[idx, 'family'] = self.genus_family_map[genus]
                    filled += 1
        
        print(f"   Filled {filled} family entries")
        print(f"   ✅ Species with family: {len(self.df[self.df['family'].notna() & (self.df['family'] != '')])}")
    
    def fill_order_from_family(self):
        """Fill missing order information based on family"""
        print("\n🔬 Step 4: Filling order information...")
        
        filled = 0
        for idx, row in self.df.iterrows():
            if (pd.isna(row['order']) or row['order'] == '') and row['family']:
                if row['family'] in self.family_order_map:
                    self.df.at[idx, 'order'] = self.family_order_map[row['family']]
                    filled += 1
        
        print(f"   Filled {filled} order entries")
        print(f"   ✅ Species with order: {len(self.df[self.df['order'].notna() & (self.df['order'] != '')])}")
    
    def standardize_habitat(self):
        """Standardize habitat values"""
        print("\n🏞️ Step 5: Standardizing habitat...")
        
        def clean_habitat(h):
            if pd.isna(h) or h == '':
                return 'Freshwater'
            h = str(h).upper()
            if 'F' in h and 'B' in h:
                return 'Freshwater, Brackish'
            elif 'F' in h and 'M' in h:
                return 'Freshwater, Marine'
            elif 'B' in h:
                return 'Brackish'
            elif 'M' in h:
                return 'Marine'
            else:
                return 'Freshwater'
        
        self.df['habitat'] = self.df['habitat'].apply(clean_habitat)
        print(f"   ✅ Standardized habitat values")
    
    def standardize_iucn_status(self):
        """Standardize IUCN status"""
        print("\n🔴 Step 6: Standardizing IUCN status...")
        
        valid_statuses = ['LC', 'NT', 'VU', 'EN', 'CR', 'EW', 'EX', 'DD', 'NE']
        
        def clean_iucn(status):
            if pd.isna(status) or status == '':
                return 'NE'
            status = str(status).strip().upper()
            if status in valid_statuses:
                return status
            return 'NE'
        
        self.df['iucn_status'] = self.df['iucn_status'].apply(clean_iucn)
        print(f"   ✅ Standardized IUCN status")
    
    def clean_size_data(self):
        """Clean and standardize size data"""
        print("\n📏 Step 7: Cleaning size data...")
        
        def clean_size(size):
            if pd.isna(size) or size == '':
                return ''
            # Remove extra spaces
            size = str(size).strip()
            # Ensure space between number and unit
            size = re.sub(r'(\d+\.?\d*)(cm|mm)', r'\1 \2', size)
            return size
        
        self.df['max_size'] = self.df['max_size'].apply(clean_size)
        print(f"   ✅ Cleaned size data")
    
    def add_common_names(self):
        """Add common English names for well-known species"""
        print("\n📝 Step 8: Adding common names...")
        
        common_names = {
            'Labeo rohita': 'Rohu',
            'Catla catla': 'Catla',
            'Cirrhinus mrigala': 'Mrigal',
            'Cirrhinus cirrhosus': 'Mrigal',
            'Channa punctata': 'Spotted Snakehead',
            'Channa striata': 'Striped Snakehead',
            'Channa marulius': 'Giant Snakehead',
            'Anabas testudineus': 'Climbing Perch',
            'Heteropneustes fossilis': 'Stinging Catfish',
            'Clarias batrachus': 'Walking Catfish',
            'Wallago attu': 'Freshwater Shark',
            'Mystus vittatus': 'Striped Dwarf Catfish',
            'Puntius sophore': 'Pool Barb',
            'Anguilla bengalensis': 'Indian Mottled Eel',
            'Notopterus notopterus': 'Bronze Featherback',
            'Chitala chitala': 'Clown Knifefish',
            'Xenentodon cancila': 'Freshwater Garfish',
            'Macrognathus pancalus': 'Barred Spiny Eel',
            'Mastacembelus armatus': 'Zig-zag Eel',
        }
        
        filled = 0
        for idx, row in self.df.iterrows():
            if (pd.isna(row['common_name']) or row['common_name'] == '') and row['scientific_name'] in common_names:
                self.df.at[idx, 'common_name'] = common_names[row['scientific_name']]
                filled += 1
        
        print(f"   Added {filled} common names")
    
    def generate_statistics(self):
        """Generate data quality statistics"""
        print("\n" + "="*70)
        print("📊 DATA QUALITY STATISTICS")
        print("="*70)
        
        total = len(self.df)
        
        print(f"\n✅ Total Species: {total}")
        print(f"📉 Removed: {self.original_count - total} ({((self.original_count - total) / self.original_count * 100):.1f}%)")
        
        print(f"\n📋 Completeness:")
        print(f"   Scientific Name: {total} (100%)")
        print(f"   Local Name: {len(self.df[self.df['local_name'].notna() & (self.df['local_name'] != '')])} ({len(self.df[self.df['local_name'].notna() & (self.df['local_name'] != '')]) / total * 100:.1f}%)")
        print(f"   Common Name: {len(self.df[self.df['common_name'].notna() & (self.df['common_name'] != '')])} ({len(self.df[self.df['common_name'].notna() & (self.df['common_name'] != '')]) / total * 100:.1f}%)")
        print(f"   Family: {len(self.df[self.df['family'].notna() & (self.df['family'] != '')])} ({len(self.df[self.df['family'].notna() & (self.df['family'] != '')]) / total * 100:.1f}%)")
        print(f"   Order: {len(self.df[self.df['order'].notna() & (self.df['order'] != '')])} ({len(self.df[self.df['order'].notna() & (self.df['order'] != '')]) / total * 100:.1f}%)")
        print(f"   Max Size: {len(self.df[self.df['max_size'].notna() & (self.df['max_size'] != '')])} ({len(self.df[self.df['max_size'].notna() & (self.df['max_size'] != '')]) / total * 100:.1f}%)")
        print(f"   IUCN Status: {len(self.df[self.df['iucn_status'].notna() & (self.df['iucn_status'] != '')])} ({len(self.df[self.df['iucn_status'].notna() & (self.df['iucn_status'] != '')]) / total * 100:.1f}%)")
        
        print(f"\n🏷️ Top 10 Families:")
        family_counts = self.df[self.df['family'].notna() & (self.df['family'] != '')]['family'].value_counts().head(10)
        for family, count in family_counts.items():
            print(f"   {family}: {count} species")
        
        print(f"\n🔴 IUCN Status Distribution:")
        iucn_counts = self.df['iucn_status'].value_counts()
        for status, count in iucn_counts.items():
            print(f"   {status}: {count} species")
    
    def save_cleaned_data(self, output_file="data/final/fish_mapping_cleaned.csv"):
        """Save cleaned data"""
        print(f"\n💾 Saving cleaned data to: {output_file}")
        self.df.to_csv(output_file, index=False)
        print(f"   ✅ Saved {len(self.df)} species")
        
        # Also save a high-quality subset (species with complete info)
        complete_df = self.df[
            (self.df['family'].notna()) & (self.df['family'] != '') &
            (self.df['order'].notna()) & (self.df['order'] != '') &
            (self.df['max_size'].notna()) & (self.df['max_size'] != '')
        ]
        
        output_complete = output_file.replace('.csv', '_complete.csv')
        complete_df.to_csv(output_complete, index=False)
        print(f"   ✅ Saved {len(complete_df)} complete records to: {output_complete}")
        
        return self.df, complete_df
    
    def run(self):
        """Run complete cleaning pipeline"""
        print("\n" + "="*70)
        print("🧹 FISH DATA CLEANING & ENRICHMENT")
        print("="*70)
        
        self.clean_scientific_names()
        self.remove_true_duplicates()
        self.fill_family_from_genus()
        self.fill_order_from_family()
        self.standardize_habitat()
        self.standardize_iucn_status()
        self.clean_size_data()
        self.add_common_names()
        self.generate_statistics()
        df, complete_df = self.save_cleaned_data()
        
        print("\n" + "="*70)
        print("✅ CLEANING COMPLETE!")
        print("="*70)
        
        print("\n🎯 Files Created:")
        print("   1. fish_mapping_cleaned.csv - All species")
        print("   2. fish_mapping_cleaned_complete.csv - Only complete records")
        
        print("\n📝 Recommended Next Steps:")
        print("   1. Review fish_mapping_cleaned_complete.csv (highest quality)")
        print("   2. Use this for your application")
        print("   3. Gradually enhance fish_mapping_cleaned.csv with manual research")
        
        return df, complete_df


if __name__ == "__main__":
    cleaner = FishDataCleaner()
    df, complete_df = cleaner.run()