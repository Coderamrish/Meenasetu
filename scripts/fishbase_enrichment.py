"""
MeenaSetu - Robust FishBase API Enrichment
Fetches professional taxonomy with better error handling and validation
"""

import pandas as pd
import requests
import time
from pathlib import Path
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class FishBaseEnricher:
    def __init__(self):
        self.master_file = Path("data/final/fish_mapping_master.csv")
        self.output_file = Path("data/final/fish_mapping_fishbase_enriched.csv")
        self.api_base = "https://fishbase.ropensci.org"
        self.cache_file = Path("data/final/fishbase_cache.csv")
        
        # Setup session with retries
        self.session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # Load cache if exists
        self.cache = {}
        if self.cache_file.exists():
            cache_df = pd.read_csv(self.cache_file)
            for _, row in cache_df.iterrows():
                self.cache[row['scientific_name']] = row.to_dict()
    
    def load_master(self):
        """Load master dataset"""
        print("\n📖 Loading master dataset...")
        self.df = pd.read_csv(self.master_file)
        print(f"   ✅ {len(self.df)} species loaded")
        
        # Clean invalid entries first
        self.clean_invalid_species()
        
        # Count what we need
        missing_family = self.df['family'].isna().sum()
        missing_order = self.df['order'].isna().sum()
        missing_size = self.df['max_size'].isna().sum()
        
        print(f"\n📊 Current gaps:")
        print(f"   • Missing family: {missing_family} ({missing_family/len(self.df)*100:.1f}%)")
        print(f"   • Missing order: {missing_order} ({missing_order/len(self.df)*100:.1f}%)")
        print(f"   • Missing size: {missing_size} ({missing_size/len(self.df)*100:.1f}%)")
    
    def is_valid_species_name(self, name):
        """Validate if this is a real species name"""
        name = str(name).strip()
        
        # Must match pattern: Genus species (optional author/year)
        pattern = r'^[A-Z][a-z]+ [a-z]+(?:\s+\([^)]+\))?'
        
        if not re.match(pattern, name):
            return False
        
        # Blacklist common non-species entries
        blacklist = [
            'Mishra et', 'Tributaries of', 'Bhakta and', 'River at',
            'Ten perennial', 'Chakraborty and', 'Major rivers', 'Patra and',
            'Karala river', 'Patra et', 'Saha and', 'Burdwan district',
            'Bandyopadhyay and', 'Sankosh rivers', 'Bera et', 'Reservoir in',
            'Kundu et', 'Tangon and', 'Mahapatra and', 'Wetlands in',
            'To elucidate', 'Two days', 'Gaighata block'
        ]
        
        for invalid in blacklist:
            if invalid.lower() in name.lower():
                return False
        
        return True
    
    def clean_invalid_species(self):
        """Remove invalid species entries"""
        print("\n🧹 Cleaning invalid entries...")
        
        before = len(self.df)
        
        # Filter valid species
        self.df = self.df[self.df['scientific_name'].apply(self.is_valid_species_name)]
        self.df = self.df.reset_index(drop=True)
        
        removed = before - len(self.df)
        
        if removed > 0:
            print(f"   ✅ Removed {removed} invalid entries")
            print(f"   ✅ {len(self.df)} valid species remaining")
    
    def clean_species_name(self, name):
        """Clean species name for API query"""
        name = str(name).strip()
        
        # Extract just "Genus species"
        match = re.match(r'^([A-Z][a-z]+ [a-z]+)', name)
        if match:
            return match.group(1)
        
        return name
    
    def query_fishbase(self, species_name):
        """Query FishBase API for a species"""
        
        # Check cache first
        if species_name in self.cache:
            return self.cache[species_name]
        
        clean_name = self.clean_species_name(species_name)
        
        try:
            # FishBase species endpoint
            url = f"{self.api_base}/species"
            params = {'Species': clean_name, 'limit': 1}
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data and len(data) > 0:
                    fish = data[0]
                    
                    result = {
                        'scientific_name': species_name,
                        'family': fish.get('Family'),
                        'order': fish.get('Order'),
                        'common_name': fish.get('FBname'),
                        'max_size': fish.get('Length'),
                        'environment': fish.get('DemersPelag'),
                        'found': True
                    }
                    
                    # Cache the result
                    self.cache[species_name] = result
                    return result
            
            # Not found
            self.cache[species_name] = {'scientific_name': species_name, 'found': False}
            return None
            
        except requests.exceptions.Timeout:
            print(f"      ⏱️  Timeout: {species_name} (skipping)")
            return None
        except requests.exceptions.ConnectionError:
            print(f"      ⚠️  Connection error: {species_name} (skipping)")
            return None
        except Exception as e:
            print(f"      ⚠️  Error for {species_name}: {str(e)[:50]}")
            return None
    
    def enrich_from_fishbase(self):
        """Enrich all species from FishBase"""
        print("\n🌊 Querying FishBase API...")
        print("   (This may take 10-15 minutes - please be patient)\n")
        
        total = len(self.df)
        enriched = 0
        not_found = 0
        skipped = 0
        errors = 0
        
        for idx, row in self.df.iterrows():
            species = row['scientific_name']
            
            # Skip if already complete
            has_family = pd.notna(row['family']) and str(row['family']).strip()
            has_order = pd.notna(row['order']) and str(row['order']).strip()
            has_size = pd.notna(row['max_size']) and str(row['max_size']).strip()
            
            if has_family and has_order and has_size:
                skipped += 1
                if (idx + 1) % 100 == 0:
                    print(f"   📊 Progress: {idx+1}/{total} (✅ {enriched} | ⏭️ {skipped} | ❌ {not_found} | ⚠️ {errors})")
                continue
            
            # Query FishBase
            result = self.query_fishbase(species)
            
            if result and result.get('found'):
                # Update fields if missing
                updated = False
                
                if not has_family and result.get('family'):
                    self.df.at[idx, 'family'] = result['family']
                    updated = True
                
                if not has_order and result.get('order'):
                    self.df.at[idx, 'order'] = result['order']
                    updated = True
                
                if not has_size and result.get('max_size'):
                    self.df.at[idx, 'max_size'] = f"{result['max_size']} cm"
                    updated = True
                
                if pd.isna(row.get('common_name')) and result.get('common_name'):
                    self.df.at[idx, 'common_name'] = result['common_name']
                    updated = True
                
                if updated:
                    enriched += 1
                    if enriched % 10 == 0:
                        print(f"   ✅ Enriched {enriched}: {species} ({result.get('family', 'N/A')})")
            elif result is None:
                errors += 1
            else:
                not_found += 1
            
            # Progress update
            if (idx + 1) % 100 == 0:
                print(f"   📊 Progress: {idx+1}/{total} (✅ {enriched} | ⏭️ {skipped} | ❌ {not_found} | ⚠️ {errors})")
            
            # Rate limiting - be nice to FishBase
            time.sleep(0.7)  # Slower to avoid timeouts
        
        print(f"\n📊 Final results:")
        print(f"   ✅ Successfully enriched: {enriched}")
        print(f"   ⏭️  Already complete: {skipped}")
        print(f"   ❌ Not found in FishBase: {not_found}")
        print(f"   ⚠️  API errors/timeouts: {errors}")
    
    def save_cache(self):
        """Save FishBase query cache"""
        print("\n💾 Saving FishBase cache...")
        
        cache_list = [v for v in self.cache.values() if isinstance(v, dict)]
        if cache_list:
            cache_df = pd.DataFrame(cache_list)
            cache_df.to_csv(self.cache_file, index=False)
            print(f"   ✅ Cached {len(cache_list)} queries")
    
    def calculate_improvements(self):
        """Calculate improvement statistics"""
        print("\n📊 Improvement Statistics:")
        
        # Load original
        original = pd.read_csv(self.master_file)
        
        fields = ['family', 'order', 'max_size', 'common_name']
        
        for field in fields:
            before = original[field].notna().sum() if field in original.columns else 0
            after = self.df[field].notna().sum() if field in self.df.columns else 0
            
            before_pct = (before / len(original)) * 100 if len(original) > 0 else 0
            after_pct = (after / len(self.df)) * 100 if len(self.df) > 0 else 0
            gained = after - before
            
            print(f"\n   {field}:")
            print(f"      Before: {before} ({before_pct:.1f}%)")
            print(f"      After:  {after} ({after_pct:.1f}%)")
            print(f"      Gained: +{gained} ({after_pct - before_pct:.1f}%)")
    
    def recalculate_quality(self):
        """Recalculate data quality scores"""
        print("\n🔢 Recalculating quality scores...")
        
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
        
        # Calculate overall quality score
        total_fields = len(complete_fields)
        avg_score = self.df['completeness_score'].mean()
        quality_score = (avg_score / total_fields) * 100
        
        print(f"\n   🎯 Overall Quality Score: {quality_score:.1f}%")
    
    def save_results(self):
        """Save enriched dataset"""
        print("\n💾 Saving enriched data...")
        
        # Backup original
        backup_file = self.master_file.parent / f"{self.master_file.stem}_pre_fishbase.csv"
        if self.master_file.exists():
            import shutil
            shutil.copy(self.master_file, backup_file)
            print(f"   ✅ Backup: {backup_file}")
        
        # Save enriched version
        self.df.to_csv(self.output_file, index=False)
        print(f"   ✅ Enriched: {self.output_file}")
        
        # Update master
        self.df.to_csv(self.master_file, index=False)
        print(f"   ✅ Updated master: {self.master_file}")
    
    def run(self):
        """Run complete FishBase enrichment"""
        print("\n" + "="*70)
        print("🌊 MeenaSetu - FishBase API Enrichment v2")
        print("="*70)
        print("\nFetching professional taxonomy from FishBase.org")
        print("This version includes better error handling & validation")
        print("="*70)
        
        self.load_master()
        self.enrich_from_fishbase()
        self.save_cache()
        self.calculate_improvements()
        self.recalculate_quality()
        self.save_results()
        
        print("\n" + "="*70)
        print("✅ FISHBASE ENRICHMENT COMPLETE!")
        print("="*70)
        
        print("\n🎯 Next Steps:")
        print("   1. Run validator: python scripts/fish_data_validator.py")
        print("   2. Review enriched data: data/final/fish_mapping_fishbase_enriched.csv")
        print("   3. For species not in FishBase, consider manual research")
        print("   4. Re-run this script later to retry failed queries")
        
        print("\n💡 Pro Tip:")
        print("   If many timeouts occurred, try running again later")
        print("   The cache prevents re-querying successful species")

if __name__ == "__main__":
    enricher = FishBaseEnricher()
    enricher.run()