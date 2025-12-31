"""
MeenaSetu - Data Validator & Quality Dashboard
Validates fish data and shows interactive quality metrics
"""

import pandas as pd
import re
from pathlib import Path
from collections import defaultdict

class FishDataValidator:
    def __init__(self, master_file=None):
        # Choose a master file: explicit param -> best CSV/JSON found in data/final -> fallback
        candidates = []
        if master_file:
            candidates.append(Path(master_file))
        # Prefer CSV variants (v2, etc.) then JSON
        candidates += sorted(Path("data/final").glob("fish_mapping_master*.csv"), key=lambda p: p.name, reverse=True)
        candidates += sorted(Path("data/final").glob("fish_mapping_master*.json"), key=lambda p: p.name, reverse=True)

        self.master_file = None
        for c in candidates:
            if c.exists():
                self.master_file = c
                break
        # final fallback path
        if self.master_file is None:
            self.master_file = Path("data/final/fish_mapping_master.csv")

        self.issues = defaultdict(list)
        self.warnings = defaultdict(list)
        
    def load_data(self):
        """Load master dataset"""
        print(f"\n📖 Loading {self.master_file}...")
        if not self.master_file.exists():
            raise FileNotFoundError(f"Master file not found: {self.master_file}")

        # Treat empty strings and common placeholders as NA
        self.df = pd.read_csv(self.master_file, na_values=["", "NA", "None", "nan", "NaN"]) 

        # Ensure columns are present and completeness_score exists
        target_cols = ['scientific_name', 'family', 'order', 'iucn_status', 'local_name', 'max_size', 'habitat']
        for c in target_cols:
            if c not in self.df.columns:
                self.df[c] = pd.NA

        if 'completeness_score' not in self.df.columns:
            self.df['completeness_score'] = self.df[target_cols].notna().sum(axis=1)

        print(f"   ✅ Loaded {len(self.df)} species\n")
    
    def validate_scientific_names(self):
        """Validate scientific name format"""
        print("🔬 Validating Scientific Names...")

        # Relaxed pattern: Genus species (optional author/year in parentheses)
        pattern = r'^[A-Z][a-zA-Z\-]+ [a-z\-]+( \(.+\d{4}\))?$'

        names = self.df['scientific_name'].fillna("")

        # Format warnings
        invalid_mask = ~names.str.match(pattern)
        for idx in self.df[invalid_mask].index:
            self.warnings['scientific_name'].append({
                'row': idx,
                'name': self.df.at[idx, 'scientific_name'],
                'issue': 'Format may be incorrect'
            })

        # Duplicates (count once per duplicated name)
        dup_counts = names.value_counts()
        duplicates = dup_counts[dup_counts > 1]
        for name, count in duplicates.items():
            self.issues['duplicates'].append({'name': name, 'count': int(count)})

        valid_count = len(self.df) - invalid_mask.sum()
        print(f"   ✅ Valid format: {valid_count}/{len(self.df)}")
        if self.warnings['scientific_name']:
            print(f"   ⚠️  Format warnings: {len(self.warnings['scientific_name'])}")
        if self.issues['duplicates']:
            print(f"   ❌ Duplicates found: {len(self.issues['duplicates'])}")
    
    def validate_taxonomy(self):
        """Validate taxonomic hierarchy"""
        print("\n🏷️  Validating Taxonomy...")
        
        # Check family-order consistency
        family_order_map = {}
        inconsistent = []
        
        for idx, row in self.df.iterrows():
            family = row.get('family')
            order = row.get('order')
            
            if pd.notna(family) and pd.notna(order):
                if family in family_order_map:
                    if family_order_map[family] != order:
                        inconsistent.append({
                            'family': family,
                            'orders': [family_order_map[family], order]
                        })
                else:
                    family_order_map[family] = order
        
        missing_family = self.df['family'].isna().sum()
        missing_order = self.df['order'].isna().sum()
        
        print(f"   📊 Family coverage: {len(self.df) - missing_family}/{len(self.df)} ({(1-missing_family/len(self.df))*100:.1f}%)")
        print(f"   📊 Order coverage: {len(self.df) - missing_order}/{len(self.df)} ({(1-missing_order/len(self.df))*100:.1f}%)")
        
        if inconsistent:
            print(f"   ⚠️  Inconsistent family-order: {len(inconsistent)}")
            self.warnings['taxonomy'] = inconsistent
    
    def validate_iucn_status(self):
        """Validate IUCN conservation status"""
        print("\n🔴 Validating IUCN Status...")

        valid_statuses = ['LC', 'NT', 'VU', 'EN', 'CR', 'EW', 'EX', 'DD', 'NE']

        invalid_count = 0
        for idx, row in self.df.iterrows():
            status = row.get('iucn_status')
            if pd.isna(status) or str(status).strip() == '':
                continue  # missing

            status_norm = str(status).strip().upper()
            if status_norm not in valid_statuses:
                invalid_count += 1
                self.issues['iucn_status'].append({
                    'row': idx,
                    'species': row.get('scientific_name'),
                    'status': status
                })

        missing = self.df['iucn_status'].isna().sum()
        valid = len(self.df) - missing - invalid_count

        print(f"   ✅ Valid: {valid}/{len(self.df)}")
        print(f"   ⚠️  Missing: {missing}")
        if invalid_count:
            print(f"   ❌ Invalid: {invalid_count}")
    
    def validate_size_data(self):
        """Validate size measurements"""
        print("\n📏 Validating Size Data...")

        for idx, row in self.df.iterrows():
            if pd.isna(row.get('max_size')):
                continue

            size = str(row.get('max_size'))
            match = re.search(r'(\d+\.?\d*)', size)
            if match:
                try:
                    value = float(match.group(1))
                except ValueError:
                    self.warnings['size'].append({
                        'row': idx,
                        'species': row.get('scientific_name'),
                        'size': size,
                        'issue': 'Cannot parse numeric value'
                    })
                    continue

                if value < 1:
                    self.warnings['size'].append({
                        'row': idx,
                        'species': row.get('scientific_name'),
                        'size': size,
                        'issue': 'Unusually small (<1cm)'
                    })
                elif value > 300:
                    self.warnings['size'].append({
                        'row': idx,
                        'species': row.get('scientific_name'),
                        'size': size,
                        'issue': 'Unusually large (>300cm)'
                    })
            else:
                self.warnings['size'].append({
                    'row': idx,
                    'species': row.get('scientific_name'),
                    'size': size,
                    'issue': 'Cannot extract numeric value'
                })

        has_size = self.df['max_size'].notna().sum()
        print(f"   📊 Size data: {has_size}/{len(self.df)} ({has_size/len(self.df)*100:.1f}%)")
        if self.warnings['size']:
            print(f"   ⚠️  Warnings: {len(self.warnings['size'])}")
    
    def validate_habitat(self):
        """Validate habitat data"""
        print("\n🏞️  Validating Habitat...")

        allowed = {'freshwater', 'brackish', 'marine'}

        for idx, row in self.df.iterrows():
            habitat = row.get('habitat')
            if pd.isna(habitat):
                continue

            # Split and normalize multi-habitat entries
            tokens = [t.strip().lower() for t in str(habitat).split(',') if t.strip()]
            bad = [t for t in tokens if t not in allowed]
            if bad:
                self.warnings['habitat'].append({
                    'row': idx,
                    'species': row.get('scientific_name'),
                    'habitat': habitat,
                    'issue': 'Non-standard habitat tokens: ' + ','.join(bad)
                })

        has_habitat = self.df['habitat'].notna().sum()
        print(f"   ✅ Present: {has_habitat}/{len(self.df)} ({has_habitat/len(self.df)*100:.1f}%)")
        if self.warnings['habitat']:
            print(f"   ⚠️  Non-standard habitats: {len(self.warnings['habitat'])}")
    
    def check_completeness(self):
        """Check data completeness"""
        print("\n📊 Completeness Analysis...")
        
        essential_fields = ['scientific_name', 'family', 'order', 'iucn_status']
        important_fields = ['local_name', 'max_size', 'habitat']
        
        print("\n   Essential Fields (should be 100%):")
        for field in essential_fields:
            if field in self.df.columns:
                coverage = (self.df[field].notna().sum() / len(self.df)) * 100
                status = "✅" if coverage > 90 else "⚠️" if coverage > 50 else "❌"
                print(f"   {status} {field}: {coverage:.1f}%")
        
        print("\n   Important Fields (target: >50%):")
        for field in important_fields:
            if field in self.df.columns:
                coverage = (self.df[field].notna().sum() / len(self.df)) * 100
                status = "✅" if coverage > 50 else "⚠️" if coverage > 25 else "❌"
                print(f"   {status} {field}: {coverage:.1f}%")
    
    def identify_priorities(self):
        """Identify species that need attention"""
        print("\n🎯 Priority Species for Enhancement...")

        # High-value species with missing data
        priority_families = ['Cyprinidae', 'Channidae', 'Bagridae', 'Sisoridae']

        priorities = []
        for idx, row in self.df.iterrows():
            if row.get('family') in priority_families:
                missing_count = 0
                for field in ['local_name', 'max_size', 'common_name']:
                    if field not in self.df.columns or pd.isna(row.get(field)):
                        missing_count += 1

                if missing_count > 0:
                    priorities.append({
                        'species': row.get('scientific_name'),
                        'family': row.get('family'),
                        'missing_fields': missing_count
                    })

        priorities.sort(key=lambda x: x['missing_fields'], reverse=True)

        print(f"\n   Found {len(priorities)} high-priority species")
        print("\n   Top 10 to enhance:")
        for i, p in enumerate(priorities[:10], 1):
            print(f"   {i}. {p['species']} ({p['family']}) - Missing {p['missing_fields']} fields")

        return priorities
    
    def save_reports(self, priorities):
        """Save validation reports"""
        print("\n💾 Saving Reports...")

        report_dir = Path("data/final/reports")
        report_dir.mkdir(parents=True, exist_ok=True)

        # Save issues
        total_issues = sum(len(v) for v in self.issues.values())
        if total_issues:
            issues_file = report_dir / "validation_issues.txt"
            with open(issues_file, 'w', encoding='utf-8') as f:
                f.write("CRITICAL ISSUES\n")
                f.write("="*70 + "\n\n")

                for issue_type, items in self.issues.items():
                    if not items:
                        continue
                    f.write(f"\n{issue_type.upper()}:\n")
                    for item in items:
                        f.write(f"  {item}\n")

            print(f"   ✅ Issues report: {issues_file}")
        else:
            print("   ✅ No critical issues to report")

        # Save warnings
        total_warnings = sum(len(v) for v in self.warnings.values())
        if total_warnings:
            warnings_file = report_dir / "validation_warnings.txt"
            with open(warnings_file, 'w', encoding='utf-8') as f:
                f.write("WARNINGS\n")
                f.write("="*70 + "\n\n")

                for warn_type, items in self.warnings.items():
                    if not items:
                        continue
                    f.write(f"\n{warn_type.upper()}:\n")
                    for item in items[:200]:  # Increase slice to capture more context
                        f.write(f"  {item}\n")
                    if len(items) > 200:
                        f.write(f"  ... and {len(items)-200} more\n")

            print(f"   ✅ Warnings report: {warnings_file}")
        else:
            print("   ✅ No warnings to report")

        # Save priorities
        priorities_file = report_dir / "enhancement_priorities.csv"
        pd.DataFrame(priorities).to_csv(priorities_file, index=False)
        print(f"   ✅ Priorities list: {priorities_file}")
    
    def generate_dashboard(self):
        """Generate quality dashboard summary"""
        print("\n" + "="*70)
        print("📊 DATA QUALITY DASHBOARD")
        print("="*70)

        total = len(self.df)

        # Overall quality score
        quality_fields = ['family', 'order', 'local_name', 'max_size', 'iucn_status']
        scores = []

        for field in quality_fields:
            if field in self.df.columns:
                coverage = self.df[field].notna().sum() / max(total, 1)
                scores.append(coverage)

        if scores:
            overall_score = sum(scores) / len(scores) * 100
        else:
            overall_score = 0.0

        print(f"\n🎯 Overall Quality Score: {overall_score:.1f}%")
        
        if overall_score >= 80:
            print("   ⭐⭐⭐⭐⭐ Excellent!")
        elif overall_score >= 60:
            print("   ⭐⭐⭐⭐ Good")
        elif overall_score >= 40:
            print("   ⭐⭐⭐ Fair - Needs improvement")
        else:
            print("   ⭐⭐ Poor - Significant work needed")
        
        # Critical vs Warning counts
        critical = sum(len(v) for v in self.issues.values())
        warnings = sum(len(v) for v in self.warnings.values())
        
        print(f"\n🚨 Issues: {critical} critical, {warnings} warnings")
        
        # Ready for production estimate
        high_quality = (self.df['completeness_score'] >= 6).sum()
        medium_quality = ((self.df['completeness_score'] >= 4) & 
                         (self.df['completeness_score'] < 6)).sum()
        
        print(f"\n📈 Production Readiness:")
        print(f"   🟢 High Quality: {high_quality} ({high_quality/total*100:.1f}%)")
        print(f"   🟡 Medium Quality: {medium_quality} ({medium_quality/total*100:.1f}%)")
        print(f"   🔴 Low Quality: {total-high_quality-medium_quality} ({(total-high_quality-medium_quality)/total*100:.1f}%)")
    
    def run(self):
        """Run complete validation"""
        print("\n" + "="*70)
        print("🔍 MeenaSetu - Data Validator")
        print("="*70)
        
        self.load_data()
        
        self.validate_scientific_names()
        self.validate_taxonomy()
        self.validate_iucn_status()
        self.validate_size_data()
        self.validate_habitat()
        self.check_completeness()
        
        priorities = self.identify_priorities()
        self.save_reports(priorities)
        self.generate_dashboard()
        
        print("\n" + "="*70)
        print("✅ VALIDATION COMPLETE!")
        print("="*70)
        
        print("\n🎯 Next Steps:")
        print("   1. Fix critical issues in validation_issues.txt")
        print("   2. Review warnings in validation_warnings.txt")
        print("   3. Enhance priority species from enhancement_priorities.csv")
        print("   4. Monitor your progress with completeness_score column")
        
        total_issues = sum(len(v) for v in self.issues.values())
        if total_issues:
            print("\n⚠️  Address critical issues before production deployment")
        else:
            print("\n✅ No critical issues! Data is production-ready")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run fish data validator')
    parser.add_argument('--master', help='Path to master file (CSV or JSON)')
    args = parser.parse_args()

    validator = FishDataValidator(master_file=args.master)
    validator.run()