"""
MeenaSetu - Smart Fish Data Merger
Intelligently merges multiple fish datasets, resolving conflicts and deduplicating
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

class SmartFishMerger:
    def __init__(self):
        self.merged_df = None
        
    def load_datasets(self):
        """Load all available fish datasets"""
        print("\n📖 Loading datasets...")
        
        datasets = {}
        
        # List of potential files to load
        files_to_try = [
            ("original", "data/final/fish_mapping.csv"),
            ("cleaned", "data/final/fish_mapping_cleaned_final.csv"),
            ("production", "data/final/fish_mapping_production.csv"),
            ("high_quality", "data/final/fish_mapping_high_quality.csv"),
            ("master", "data/final/fish_mapping_master.csv"),
            ("enriched", "data/final/fish_mapping_enriched.csv"),
        ]
        
        for name, filepath in files_to_try:
            try:
                df = pd.read_csv(filepath)
                datasets[name] = df
                print(f"   ✅ {name:15s}: {len(df):4d} species - {filepath}")
            except FileNotFoundError:
                print(f"   ⏭️  {name:15s}: Not found - {filepath}")
        
        if not datasets:
            print("\n   ❌ No datasets found! Please run the cleaner first.")
            return None
        
        print(f"\n   📊 Total datasets loaded: {len(datasets)}")
        return datasets
    
    def choose_best_value(self, values):
        """Choose the best value from multiple sources"""
        # Filter out NaN and empty strings
        valid_values = [v for v in values if pd.notna(v) and v != '']
        
        if not valid_values:
            return ''
        
        # If all values are the same, return it
        if len(set(valid_values)) == 1:
            return valid_values[0]
        
        # Prefer longer, more complete values
        return max(valid_values, key=lambda x: len(str(x)))
    
    def merge_species_data(self, species_name, records):
        """Merge multiple records of the same species, choosing best data"""
        if len(records) == 1:
            return records[0]
        
        # Create merged record
        merged = {
            'scientific_name': species_name,
            'common_name': '',
            'local_name': '',
            'family': '',
            'order': '',
            'max_size': '',
            'habitat': '',
            'iucn_status': '',
            'human_use': '',
            'distribution': '',
            'conservation_notes': '',
            'source': []
        }
        
        # For each field, choose the best value
        for record in records:
            for field in merged.keys():
                if field == 'source':
                    if 'source' in record and pd.notna(record['source']):
                        merged['source'].append(str(record['source']))
                else:
                    if field in record:
                        current_val = merged[field]
                        new_val = record[field]
                        
                        # Choose better value
                        if pd.isna(current_val) or current_val == '':
                            merged[field] = new_val if pd.notna(new_val) else ''
                        elif pd.notna(new_val) and new_val != '':
                            # Prefer longer/more complete values
                            if len(str(new_val)) > len(str(current_val)):
                                merged[field] = new_val
        
        # Combine sources
        merged['source'] = ', '.join(set([s for s in merged['source'] if s]))
        
        return merged
    
    def merge_all_datasets(self, datasets):
        """Merge all datasets intelligently"""
        print("\n🔄 Merging datasets...")
        
        # Combine all datasets
        all_records = []
        
        for name, df in datasets.items():
            # Add source column if not present
            if 'source' not in df.columns:
                df['source'] = name
            
            all_records.append(df)
        
        # Concatenate all
        combined = pd.concat(all_records, ignore_index=True)
        print(f"   📊 Combined records: {len(combined)}")
        
        # Group by scientific name and merge
        print(f"   🔍 Identifying unique species...")
        
        species_groups = combined.groupby('scientific_name')
        merged_records = []
        
        for species_name, group in species_groups:
            records = group.to_dict('records')
            merged = self.merge_species_data(species_name, records)
            merged_records.append(merged)
        
        self.merged_df = pd.DataFrame(merged_records)
        
        print(f"   ✅ Unique species after merge: {len(self.merged_df)}")
        print(f"   📉 Duplicates removed: {len(combined) - len(self.merged_df)}")
        
        return self.merged_df
    
    def calculate_completeness(self):
        """Calculate completeness score for each species"""
        print("\n📊 Calculating completeness scores...")
        
        def get_score(row):
            score = 0
            fields = ['scientific_name', 'local_name', 'family', 'order', 'max_size']
            for field in fields:
                if field in row and pd.notna(row[field]) and row[field] != '':
                    score += 1
            return score
        
        self.merged_df['completeness_score'] = self.merged_df.apply(get_score, axis=1)
        
        # Quality tiers
        high = len(self.merged_df[self.merged_df['completeness_score'] >= 4])
        medium = len(self.merged_df[self.merged_df['completeness_score'] == 3])
        low = len(self.merged_df[self.merged_df['completeness_score'] < 3])
        
        print(f"   🟢 High Quality (≥4/5): {high} ({high/len(self.merged_df)*100:.1f}%)")
        print(f"   🟡 Medium Quality (3/5): {medium} ({medium/len(self.merged_df)*100:.1f}%)")
        print(f"   🔴 Low Quality (<3/5): {low} ({low/len(self.merged_df)*100:.1f}%)")
    
    def prioritize_records(self):
        """Sort by quality and relevance"""
        print("\n📌 Prioritizing records...")
        
        # Sort by completeness score (desc), then alphabetically
        self.merged_df = self.merged_df.sort_values(
            ['completeness_score', 'scientific_name'],
            ascending=[False, True]
        ).reset_index(drop=True)
        
        print(f"   ✅ Sorted by quality and name")
    
    def generate_reports(self):
        """Generate comprehensive reports"""
        print("\n" + "="*70)
        print("📊 MERGE REPORT")
        print("="*70)
        
        total = len(self.merged_df)
        
        print(f"\n✅ Total Unique Species: {total}")
        
        print(f"\n📋 Field Completeness:")
        fields = ['scientific_name', 'local_name', 'common_name', 'family', 'order', 
                  'max_size', 'habitat', 'iucn_status', 'human_use']
        
        for field in fields:
            if field in self.merged_df.columns:
                count = len(self.merged_df[self.merged_df[field].notna() & (self.merged_df[field] != '')])
                pct = count / total * 100
                bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
                print(f"   {field:20s}: {count:4d}/{total} ({pct:5.1f}%) {bar}")
        
        print(f"\n🏷️  Top 10 Families by Species Count:")
        if 'family' in self.merged_df.columns:
            family_counts = self.merged_df[
                self.merged_df['family'].notna() & (self.merged_df['family'] != '')
            ]['family'].value_counts().head(10)
            
            for family, count in family_counts.items():
                print(f"   {family:25s}: {count:3d} species")
        
        print(f"\n🌍 Distribution by Order:")
        if 'order' in self.merged_df.columns:
            order_counts = self.merged_df[
                self.merged_df['order'].notna() & (self.merged_df['order'] != '')
            ]['order'].value_counts()
            
            for order, count in order_counts.items():
                print(f"   {order:25s}: {count:3d} species")
    
    def save_outputs(self):
        """Save all output files"""
        print("\n💾 Saving merged datasets...")
        
        Path("data/final/merged").mkdir(parents=True, exist_ok=True)
        
        # 1. Master merged file (all species)
        output_master = "data/final/merged/fish_mapping_merged_master.csv"
        self.merged_df.to_csv(output_master, index=False)
        print(f"   ✅ Master file: {output_master}")
        print(f"      ({len(self.merged_df)} species)")
        
        # 2. High quality subset (completeness ≥ 4)
        high_quality = self.merged_df[self.merged_df['completeness_score'] >= 4].copy()
        output_hq = "data/final/merged/fish_mapping_merged_high_quality.csv"
        high_quality.to_csv(output_hq, index=False)
        print(f"   ✅ High quality: {output_hq}")
        print(f"      ({len(high_quality)} species)")
        
        # 3. Production ready (complete taxonomy + size)
        production = self.merged_df[
            (self.merged_df['family'].notna()) & (self.merged_df['family'] != '') &
            (self.merged_df['order'].notna()) & (self.merged_df['order'] != '') &
            (self.merged_df['max_size'].notna()) & (self.merged_df['max_size'] != '')
        ].copy()
        output_prod = "data/final/merged/fish_mapping_merged_production.csv"
        production.to_csv(output_prod, index=False)
        print(f"   ✅ Production ready: {output_prod}")
        print(f"      ({len(production)} species - ready for app!)")
        
        # 4. Top 300 species by quality
        top_300 = self.merged_df.head(300).copy()
        output_top = "data/final/merged/fish_mapping_top_300.csv"
        top_300.to_csv(output_top, index=False)
        print(f"   ✅ Top 300: {output_top}")
        print(f"      (Best quality species)")
        
        # 5. Species needing work
        needs_work = self.merged_df[self.merged_df['completeness_score'] < 3].copy()
        output_needs = "data/final/merged/needs_enhancement.csv"
        needs_work.to_csv(output_needs, index=False)
        print(f"   ✅ Needs work: {output_needs}")
        print(f"      ({len(needs_work)} species)")
        
        # 6. Summary statistics
        summary = {
            'total_species': len(self.merged_df),
            'high_quality': len(high_quality),
            'production_ready': len(production),
            'needs_enhancement': len(needs_work),
            'family_coverage': f"{len(self.merged_df[self.merged_df['family'].notna() & (self.merged_df['family'] != '')]) / len(self.merged_df) * 100:.1f}%",
            'order_coverage': f"{len(self.merged_df[self.merged_df['order'].notna() & (self.merged_df['order'] != '')]) / len(self.merged_df) * 100:.1f}%",
        }
        
        summary_df = pd.DataFrame([summary])
        output_summary = "data/final/merged/merge_summary.csv"
        summary_df.to_csv(output_summary, index=False)
        print(f"   ✅ Summary: {output_summary}")
        
        return production, high_quality
    
    def run(self):
        """Run complete merge pipeline"""
        print("\n" + "="*70)
        print("🔀 SMART FISH DATA MERGER")
        print("="*70)
        print("\nIntelligently combines multiple datasets, choosing best values")
        print("Resolves conflicts and removes duplicates")
        
        # Load datasets
        datasets = self.load_datasets()
        if datasets is None:
            return None, None
        
        # Merge
        self.merge_all_datasets(datasets)
        
        # Process
        self.calculate_completeness()
        self.prioritize_records()
        
        # Report
        self.generate_reports()
        
        # Save
        production, high_quality = self.save_outputs()
        
        print("\n" + "="*70)
        print("✅ MERGE COMPLETE!")
        print("="*70)
        
        print("\n🎯 RECOMMENDED FILES:")
        print(f"   🥇 fish_mapping_merged_production.csv")
        print(f"      → {len(production)} complete species - USE THIS IN YOUR APP!")
        print(f"   🥈 fish_mapping_top_300.csv")
        print(f"      → 300 best quality species - Perfect for testing")
        print(f"   🥉 fish_mapping_merged_master.csv")
        print(f"      → {len(self.merged_df)} total species - Complete dataset")
        
        print("\n📝 Next Steps:")
        print("   1. Copy fish_mapping_merged_production.csv to your app directory")
        print("   2. Test with fish_mapping_top_300.csv first")
        print("   3. Review needs_enhancement.csv for gaps")
        print("   4. You're ready to build your fish identification features! 🚀")
        
        return production, self.merged_df


if __name__ == "__main__":
    merger = SmartFishMerger()
    production, master = merger.run()