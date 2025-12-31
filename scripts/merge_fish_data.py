"""
MeenaSetu - Intelligent Fish Data Merger
Merges auto-generated data with manual entries, prioritizing quality
"""

import pandas as pd
import os
from pathlib import Path

class FishDataMerger:
    def __init__(self):
        self.base_dir = Path("data/final")
        self.auto_file = self.base_dir / "fish_mapping_cleaned.csv"
        self.manual_file = self.base_dir / "fish_mapping.csv"
        self.output_file = self.base_dir / "fish_mapping_master.csv"
        
    def load_data(self):
        """Load both datasets"""
        print("\n📖 Loading datasets...")
        
        # Load auto-generated data
        if self.auto_file.exists():
            self.auto_df = pd.read_csv(self.auto_file)
            print(f"   ✅ Auto-generated: {len(self.auto_df)} species")
        else:
            print(f"   ❌ {self.auto_file} not found!")
            self.auto_df = pd.DataFrame()
        
        # Load manual data
        if self.manual_file.exists():
            self.manual_df = pd.read_csv(self.manual_file)
            print(f"   ✅ Manual entries: {len(self.manual_df)} species")
        else:
            print(f"   ⚠️  No manual file found, will use auto-generated only")
            self.manual_df = pd.DataFrame()
    
    def merge_intelligent(self):
        """Merge datasets with intelligent conflict resolution"""
        print("\n🔄 Merging datasets...")
        
        if self.manual_df.empty:
            print("   ℹ️  No manual data to merge, using auto-generated")
            self.merged_df = self.auto_df.copy()
            return
        
        if self.auto_df.empty:
            print("   ℹ️  No auto-generated data, using manual")
            self.merged_df = self.manual_df.copy()
            return
        
        # Start with auto-generated as base
        self.merged_df = self.auto_df.copy()
        
        # Track statistics
        added = 0
        updated = 0
        conflicts = []
        
        # Process each manual entry
        for idx, manual_row in self.manual_df.iterrows():
            sci_name = manual_row['scientific_name']
            
            # Check if species exists in auto-generated
            mask = self.merged_df['scientific_name'] == sci_name
            
            if mask.any():
                # Species exists - merge fields intelligently
                auto_idx = self.merged_df[mask].index[0]
                
                # Priority: Manual > Auto for most fields
                priority_fields = ['local_name', 'common_name', 'family', 'order', 
                                 'habitat', 'max_size', 'iucn_status', 'uses', 'notes']
                
                for field in priority_fields:
                    if field in manual_row and field in self.merged_df.columns:
                        manual_val = manual_row[field]
                        auto_val = self.merged_df.at[auto_idx, field]
                        
                        # Use manual if not empty
                        if pd.notna(manual_val) and str(manual_val).strip():
                            if pd.notna(auto_val) and str(auto_val).strip():
                                if str(manual_val).lower() != str(auto_val).lower():
                                    conflicts.append({
                                        'species': sci_name,
                                        'field': field,
                                        'manual': manual_val,
                                        'auto': auto_val
                                    })
                            self.merged_df.at[auto_idx, field] = manual_val
                            updated += 1
            else:
                # New species from manual - add it
                self.merged_df = pd.concat([self.merged_df, manual_row.to_frame().T], 
                                          ignore_index=True)
                added += 1
        
        print(f"   ✅ Added {added} new species from manual")
        print(f"   ✅ Updated {updated} fields from manual entries")
        
        if conflicts:
            print(f"   ⚠️  Found {len(conflicts)} conflicts (manual took priority)")
            self._save_conflicts(conflicts)
    
    def _save_conflicts(self, conflicts):
        """Save conflict report"""
        conflict_file = self.base_dir / "merge_conflicts.csv"
        pd.DataFrame(conflicts).to_csv(conflict_file, index=False)
        print(f"      📝 Saved conflict report to: {conflict_file}")
    
    def add_data_source_column(self):
        """Add column indicating data source"""
        print("\n🏷️  Adding data source tracking...")
        
        # Mark sources
        self.merged_df['data_source'] = 'auto'
        
        if not self.manual_df.empty:
            for sci_name in self.manual_df['scientific_name']:
                mask = self.merged_df['scientific_name'] == sci_name
                if mask.any():
                    current = self.merged_df.loc[mask, 'data_source'].iloc[0]
                    self.merged_df.loc[mask, 'data_source'] = 'manual+auto' if current == 'auto' else 'manual'
    
    def quality_check(self):
        """Perform quality checks"""
        print("\n🔍 Quality Check...")
        
        total = len(self.merged_df)
        
        # Completeness check
        complete_fields = ['scientific_name', 'local_name', 'family', 'order', 
                          'habitat', 'max_size', 'iucn_status']
        
        completeness = {}
        for field in complete_fields:
            if field in self.merged_df.columns:
                non_null = self.merged_df[field].notna().sum()
                completeness[field] = (non_null / total * 100)
        
        print("\n   📊 Data Completeness:")
        for field, pct in completeness.items():
            print(f"      {field}: {pct:.1f}%")
        
        # Find high-quality records (>=80% complete)
        self.merged_df['completeness_score'] = 0
        for field in complete_fields:
            if field in self.merged_df.columns:
                self.merged_df['completeness_score'] += self.merged_df[field].notna().astype(int)
        
        high_quality = (self.merged_df['completeness_score'] >= 6).sum()
        print(f"\n   ⭐ High Quality Records (≥6/7 fields): {high_quality} ({high_quality/total*100:.1f}%)")
        
        return high_quality
    
    def save_outputs(self):
        """Save merged data in multiple formats"""
        print("\n💾 Saving outputs...")
        
        # Save master file
        self.merged_df.to_csv(self.output_file, index=False)
        print(f"   ✅ Master file: {self.output_file}")
        print(f"      Total species: {len(self.merged_df)}")
        
        # Save high-quality subset
        high_quality_df = self.merged_df[self.merged_df['completeness_score'] >= 6].copy()
        hq_file = self.base_dir / "fish_mapping_high_quality.csv"
        high_quality_df.to_csv(hq_file, index=False)
        print(f"   ✅ High quality: {hq_file}")
        print(f"      Records: {len(high_quality_df)}")
        
        # Save JSON version
        json_file = self.base_dir / "fish_mapping_master.json"
        self.merged_df.to_json(json_file, orient='records', indent=2)
        print(f"   ✅ JSON version: {json_file}")
        
        # Save by family for easier lookup
        if 'family' in self.merged_df.columns:
            by_family_dir = self.base_dir / "by_family"
            by_family_dir.mkdir(exist_ok=True)
            
            for family in self.merged_df['family'].dropna().unique():
                family_df = self.merged_df[self.merged_df['family'] == family]
                family_file = by_family_dir / f"{family}.csv"
                family_df.to_csv(family_file, index=False)
            
            print(f"   ✅ By family: {by_family_dir}/ ({len(self.merged_df['family'].unique())} families)")
    
    def generate_statistics(self):
        """Generate detailed statistics"""
        print("\n" + "="*70)
        print("📊 FINAL STATISTICS")
        print("="*70)
        
        total = len(self.merged_df)
        print(f"\n✅ Total Species: {total}")
        
        # By data source
        if 'data_source' in self.merged_df.columns:
            print("\n📚 By Data Source:")
            for source, count in self.merged_df['data_source'].value_counts().items():
                print(f"   {source}: {count} ({count/total*100:.1f}%)")
        
        # Top families
        if 'family' in self.merged_df.columns:
            print("\n🏷️  Top 15 Families:")
            for family, count in self.merged_df['family'].value_counts().head(15).items():
                print(f"   {family}: {count} species")
        
        # IUCN status
        if 'iucn_status' in self.merged_df.columns:
            print("\n🔴 IUCN Conservation Status:")
            for status, count in self.merged_df['iucn_status'].value_counts().items():
                print(f"   {status}: {count} species")
        
        # Habitat distribution
        if 'habitat' in self.merged_df.columns:
            print("\n🏞️  Habitat Distribution:")
            for habitat, count in self.merged_df['habitat'].value_counts().head(10).items():
                print(f"   {habitat}: {count} species")
    
    def run(self):
        """Run complete merge process"""
        print("\n" + "="*70)
        print("🐟 MeenaSetu - Fish Data Merger")
        print("="*70)
        
        self.load_data()
        self.merge_intelligent()
        self.add_data_source_column()
        hq_count = self.quality_check()
        self.save_outputs()
        self.generate_statistics()
        
        print("\n" + "="*70)
        print("✅ MERGE COMPLETE!")
        print("="*70)
        
        print("\n🎯 Next Steps:")
        print("   1. Review: data/final/fish_mapping_master.csv")
        print("   2. Use high-quality subset for your app")
        print("   3. Check merge_conflicts.csv if it exists")
        print("   4. Continue enriching data manually")
        
        print(f"\n💡 You now have {len(self.merged_df)} species ready for MeenaSetu!")
        print(f"   {hq_count} are high-quality (≥6/7 fields complete)")

if __name__ == "__main__":
    merger = FishDataMerger()
    merger.run()