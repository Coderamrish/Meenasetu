# Fish Mapping Data Entry Guide

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
