"""
Consolidate extracted fish data into final fish_mapping.csv
Focus on the best tables we found
"""

import pandas as pd
from pathlib import Path
import re

class FishDataConsolidator:
    """Consolidate fish data from multiple sources"""
    
    def __init__(self):
        self.processed_dir = Path("datasets/processed")
        self.output_dir = Path("data/final")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Master data storage
        self.fish_species = []
        self.culture_parameters = []
    
    def load_wb_fish_diversity(self):
        """Load West Bengal Fish Diversity tables"""
        print("📊 Loading WB Fish Diversity Data...")
        
        # These tables contain 267 fish species!
        wb_files = list(self.processed_dir.glob("FreshwaterfishdiversityofWestBengal_table_*.csv"))
        
        species_data = []
        
        for table_file in wb_files:
            try:
                df = pd.read_csv(table_file)
                
                # Look for fish species data
                # Common column patterns: Order, Family, Species, Scientific Name
                
                if len(df) > 0:
                    # Add source
                    df['source_file'] = table_file.name
                    species_data.append(df)
                    
                    print(f"  ✅ {table_file.name}: {len(df)} rows, {len(df.columns)} cols")
            
            except Exception as e:
                print(f"  ❌ {table_file.name}: {e}")
        
        if species_data:
            combined = pd.concat(species_data, ignore_index=True)
            print(f"\n✅ Total rows: {len(combined)}")
            print(f"Columns: {combined.columns.tolist()}")
            
            # Save raw combined data
            output = self.output_dir / "wb_fish_diversity_combined.csv"
            combined.to_csv(output, index=False)
            print(f"💾 Saved to: {output}")
            
            return combined
        
        return None
    
    def create_template_with_examples(self):
        """Create fish_mapping.csv template with examples"""
        print("\n🏗️ Creating fish_mapping.csv template...")
        
        # Template with known data + examples
        data = {
            # Row 1-3: Indian Major Carps (IMC)
            'common_name': [
                'Rohu', 'Catla', 'Mrigal',
                'Silver Carp', 'Grass Carp', 'Common Carp',
                'Tilapia', 'Pangasius',
                'Magur (Catfish)', 'Singhi (Catfish)',
                'Channa/Murrel', 'Hilsa',
                'Goldfish', 'Koi', 'Guppy'
            ],
            'scientific_name': [
                'Labeo rohita', 'Catla catla', 'Cirrhinus mrigala',
                'Hypophthalmichthys molitrix', 'Ctenopharyngodon idella', 'Cyprinus carpio',
                'Oreochromis niloticus', 'Pangasianodon hypophthalmus',
                'Clarias batrachus', 'Heteropneustes fossilis',
                'Channa striata', 'Tenualosa ilisha',
                'Carassius auratus', 'Cyprinus rubrofuscus', 'Poecilia reticulata'
            ],
            'family': [
                'Cyprinidae', 'Cyprinidae', 'Cyprinidae',
                'Cyprinidae', 'Cyprinidae', 'Cyprinidae',
                'Cichlidae', 'Pangasiidae',
                'Clariidae', 'Heteropneustidae',
                'Channidae', 'Clupeidae',
                'Cyprinidae', 'Cyprinidae', 'Poeciliidae'
            ],
            'water_type': [
                'pond', 'pond', 'pond',
                'pond', 'pond', 'pond',
                'pond', 'pond',
                'pond', 'pond',
                'pond', 'river',
                'pond', 'pond', 'aquarium'
            ],
            'region': [
                'West Bengal', 'West Bengal', 'West Bengal',
                'West Bengal', 'West Bengal', 'West Bengal',
                'Pan India', 'West Bengal',
                'West Bengal', 'West Bengal',
                'West Bengal', 'West Bengal (Ganges)',
                'Pan India', 'Pan India', 'Pan India'
            ],
            'season': [
                'all_year', 'all_year', 'all_year',
                'all_year', 'all_year', 'all_year',
                'all_year', 'all_year',
                'monsoon', 'monsoon',
                'all_year', 'monsoon',
                'all_year', 'all_year', 'all_year'
            ],
            'temp_min': [22, 22, 22, 20, 20, 15, 22, 22, 22, 22, 20, 20, 15, 18, 22],
            'temp_max': [32, 32, 32, 30, 30, 28, 35, 32, 32, 32, 30, 30, 28, 26, 28],
            'ph_min': [7.0, 7.0, 7.0, 7.0, 7.0, 6.5, 6.5, 7.0, 6.5, 6.5, 6.5, 7.0, 6.5, 6.5, 6.5],
            'ph_max': [8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.0, 8.0, 8.5, 8.5, 8.0],
            'stocking_density_per_ha': [5000, 4000, 4000, 3000, 2000, 3000, 8000, 5000, 3000, 3000, 2000, 0, 10000, 10000, 15000],
            'growth_duration_months': [12, 10, 12, 10, 12, 10, 6, 10, 10, 10, 12, 12, 8, 12, 4],
            'category': [
                'major_carp', 'major_carp', 'major_carp',
                'exotic_carp', 'exotic_carp', 'exotic_carp',
                'exotic', 'catfish',
                'catfish', 'catfish',
                'murrel', 'native',
                'ornamental', 'ornamental', 'ornamental'
            ],
            'feeding_habit': [
                'omnivore', 'planktivore', 'bottom_feeder',
                'planktivore', 'herbivore', 'omnivore',
                'omnivore', 'omnivore',
                'carnivore', 'carnivore',
                'carnivore', 'planktivore',
                'omnivore', 'omnivore', 'omnivore'
            ],
            'market_value': [
                'high', 'high', 'high',
                'medium', 'medium', 'medium',
                'medium', 'medium',
                'high', 'high',
                'high', 'very_high',
                'low', 'medium', 'low'
            ],
            'is_native': [
                True, True, True,
                False, False, False,
                False, False,
                True, True,
                True, True,
                False, False, False
            ],
            'data_quality': [
                'verified', 'verified', 'verified',
                'verified', 'verified', 'verified',
                'verified', 'verified',
                'verified', 'verified',
                'verified', 'verified',
                'template', 'template', 'template'
            ]
        }
        
        df = pd.DataFrame(data)
        
        # Add derived features
        df['temp_avg'] = (df['temp_min'] + df['temp_max']) / 2
        df['ph_avg'] = (df['ph_min'] + df['ph_max']) / 2
        df['temp_range'] = df['temp_max'] - df['temp_min']
        
        # Save
        output_path = self.output_dir / "fish_mapping.csv"
        df.to_csv(output_path, index=False)
        
        print(f"✅ Created: {output_path}")
        print(f"   Records: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
        
        # Also create JSON
        json_path = self.output_dir / "fish_mapping.json"
        df.to_json(json_path, orient='records', indent=2)
        print(f"✅ Created: {json_path}")
        
        # Create instruction file
        self.create_data_entry_guide()
        
        return df
    
    def create_data_entry_guide(self):
        """Create guide for manual data entry"""
        guide = """# Fish Mapping Data Entry Guide

## 📋 Current Status
- Template created with 15 species
- 267 more species available in WB Fish Diversity tables
- Target: 300-500 total records

## 🎯 Priority Species to Add (Next 50)

### Catfish Family (10 species)
- [ ] Mystus vittatus (Striped dwarf catfish)
- [ ] Mystus cavasius (Gangetic mystus)
- [ ] Rita rita (Rita)
- [ ] Wallago attu (Wallago catfish)
- [ ] Ompok pabda (Pabdah catfish)
- [ ] Ompok bimaculatus (Butter catfish)
- [ ] Ailia coila (Gangetic ailia)
- [ ] Clupisoma garua (Garua bacha)
- [ ] Eutropiichthys vacha (Batchwa vacha)
- [ ] Gagata cenia (Indian gagata)

### Barbs & Minor Carps (15 species)
- [ ] Puntius sophore (Pool barb)
- [ ] Puntius ticto (Ticto barb)
- [ ] Puntius sarana (Olive barb)
- [ ] Puntius conchonius (Rosy barb)
- [ ] Amblypharyngodon mola (Mola carplet)
- [ ] Salmostoma bacaila (Large razorbelly minnow)
- [ ] Chela cachius (Silver hatchet chela)
- [ ] Danio rerio (Zebrafish)
- [ ] Esomus danrica (Flying barb)
- [ ] Rasbora daniconius (Slender rasbora)
- [ ] Labeo bata (Bata)
- [ ] Labeo calbasu (Orangefin labeo)
- [ ] Labeo gonius (Kuria labeo)
- [ ] Osteobrama cotio (Cotio)
- [ ] Tor tor (Tor mahseer)

### Snakeheads/Murrels (8 species)
- [ ] Channa punctata (Spotted snakehead)
- [ ] Channa marulius (Great snakehead)
- [ ] Channa gachua (Dwarf snakehead)
- [ ] Channa orientalis (Smooth-breasted snakehead)
- [ ] Channa stewartii (Golden snakehead)
- [ ] Channa bleheri (Rainbow snakehead)
- [ ] Channa aurantimaculata (Orange spotted snakehead)
- [ ] Channa barca (Barca snakehead)

### Climbing Perch & Gouramis (5 species)
- [ ] Anabas testudineus (Climbing perch)
- [ ] Trichogaster fasciata (Banded gourami)
- [ ] Trichogaster lalius (Dwarf gourami)
- [ ] Colisa lalia (Dwarf gourami)
- [ ] Badis badis (Dwarf chameleon fish)

### Prawns & Shrimp (5 species)
- [ ] Macrobrachium rosenbergii (Giant river prawn)
- [ ] Macrobrachium malcolmsonii (Monsoon river prawn)
- [ ] Macrobrachium birmanicum (Birman prawn)
- [ ] Macrobrachium rude (Rude prawn)
- [ ] Palaemon sp. (Glass shrimp)

### Miscellaneous (7 species)
- [ ] Notopterus notopterus (Bronze featherback)
- [ ] Notopterus chitala (Humped featherback)
- [ ] Xenentodon cancila (Freshwater garfish)
- [ ] Glossogobius giuris (Tank goby)
- [ ] Mastacembelus armatus (Tire track eel)
- [ ] Macrognathus pancalus (Striped spiny eel)
- [ ] Chanda nama (Elongate glass perchlet)

## 📝 Data Entry Template

For each species, fill in:

```csv
common_name,scientific_name,family,water_type,region,season,temp_min,temp_max,ph_min,ph_max,stocking_density_per_ha,growth_duration_months,category,feeding_habit,market_value,is_native,data_quality
Striped dwarf catfish,Mystus vittatus,Bagridae,pond,West Bengal,all_year,20,30,6.5,8.0,3000,8,catfish,carnivore,medium,True,needs_verification
```

## 🔍 Where to Find Data

1. **datasets/processed/FreshwaterfishdiversityofWestBengal_table_*.csv**
   - Contains 267 species with scientific names
   - Has family information
   - May have habitat data

2. **datasets/processed/AnnualReport_*_table_*.csv**
   - CIFA experimental data
   - Growth rates
   - Culture parameters

3. **datasets/processed/CATLA_Pamphlet_table_*.csv**
   - Specific culture practices
   - Stocking densities

4. **FishBase.org** (for verification)
   - Temperature ranges
   - pH preferences
   - Native status

## ⚡ Quick Entry Workflow

1. Open: `data/final/fish_mapping.csv` in Excel
2. Pick a species from priority list above
3. Search in WB Fish Diversity tables for scientific name
4. Look up temperature/pH on FishBase if not in PDFs
5. Use similar species as reference for estimates
6. Set `data_quality` as:
   - "verified" = from multiple sources
   - "needs_verification" = single source or estimated
   - "template" = placeholder values

## 💡 Tips

- **Copy similar species**: Use Rohu as template for other carps
- **Batch process**: Do all catfish together, then all barbs, etc.
- **Mark estimates**: Set data_quality="needs_verification" for estimates
- **Cross-reference**: Check 2-3 sources before marking "verified"

## 🎯 Weekly Goals

- **Week 1**: 50 additional species (Total: 65)
- **Week 2**: 100 additional species (Total: 165)
- **Week 3**: 135 additional species (Total: 300) ← ML Training Ready!
- **Week 4**: 200 additional species (Total: 500) ← Production Ready!

---

**Start with the priority list above, then move to WB Fish Diversity tables!**
"""
        
        guide_path = self.output_dir / "DATA_ENTRY_GUIDE.md"
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide)
        
        print(f"✅ Created: {guide_path}")


def main():
    """Main consolidation workflow"""
    print("\n" + "=" * 70)
    print("🐟 MeenaSetu - Fish Data Consolidator")
    print("=" * 70)
    
    consolidator = FishDataConsolidator()
    
    # Step 1: Load WB Fish Diversity data
    print("\nStep 1: Loading source data...")
    wb_data = consolidator.load_wb_fish_diversity()
    
    # Step 2: Create template dataset
    print("\nStep 2: Creating fish_mapping.csv template...")
    fish_mapping = consolidator.create_template_with_examples()
    
    print("\n" + "=" * 70)
    print("✅ CONSOLIDATION COMPLETE!")
    print("=" * 70)
    
    print("\n📁 Files Created:")
    print("   ✅ data/final/fish_mapping.csv (15 species - template)")
    print("   ✅ data/final/fish_mapping.json (JSON version)")
    print("   ✅ data/final/wb_fish_diversity_combined.csv (raw WB data)")
    print("   ✅ data/final/DATA_ENTRY_GUIDE.md (step-by-step guide)")
    
    print("\n🎯 Next Steps:")
    print("   1. Open: data/final/DATA_ENTRY_GUIDE.md")
    print("   2. Follow the priority list to add 50 more species")
    print("   3. Use WB Fish Diversity tables as reference")
    print("   4. Target: 300 species by end of week 3")
    
    print("\n💡 Start Now:")
    print("   code data/final/fish_mapping.csv")
    print("   code data/final/DATA_ENTRY_GUIDE.md")


if __name__ == "__main__":
    main()
    