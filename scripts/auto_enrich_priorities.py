"""Auto-enrich top priority species using internal heuristics.

Strategy:
- Read enhancement_priorities.csv
- For the top N species (default 20), fill missing fields using:
  - For local_name: mode of local_name within same genus, then family
  - For max_size: median numeric value within same genus, then family
  - For habitat: mode of habitat within same genus, then family
- Save updated master as fish_mapping_master_v3.csv and write a small changes report.
"""

import pandas as pd
from pathlib import Path
import re

MASTER_V2 = Path('data/final/fish_mapping_master_v2.csv')
MASTER_V3 = Path('data/final/fish_mapping_master_v3.csv')
PRIORITIES = Path('data/final/reports/enhancement_priorities.csv')
REPORT = Path('data/final/reports/auto_enrichment_report.csv')

N = 20

def parse_genus(sci_name):
    if pd.isna(sci_name):
        return None
    parts = str(sci_name).split()
    if not parts:
        return None
    return parts[0].strip()


def extract_numeric_size(x):
    if pd.isna(x):
        return None
    s = str(x)
    m = re.search(r"(\d+\.?\d*)", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def find_mode(series):
    s = series.dropna()
    if s.empty:
        return None
    return s.mode().iloc[0]


def median_numeric(series):
    nums = [v for v in series.map(extract_numeric_size).dropna()]
    if not nums:
        return None
    return float(pd.Series(nums).median())


def enrich_top(n=N):
    df = pd.read_csv(MASTER_V2, na_values=['', 'NA', 'None', 'nan', 'NaN'])
    if not PRIORITIES.exists():
        raise FileNotFoundError('Priorities file not found: ' + str(PRIORITIES))
    pr = pd.read_csv(PRIORITIES)

    changes = []
    top = pr.head(n)

    for _, row in top.iterrows():
        sci = row.get('species')
        if pd.isna(sci):
            continue
        matches = df[df['scientific_name'] == sci]
        if len(matches) == 0:
            continue
        idx = matches.index[0]

        genus = parse_genus(sci)
        family = df.at[idx, 'family'] if 'family' in df.columns else None

        # local_name
        if 'local_name' in df.columns and pd.isna(df.at[idx, 'local_name']):
            val = None
            if genus:
                val = find_mode(df[df['scientific_name'].str.startswith(genus + ' ', na=False)]['local_name'])
            if not val and pd.notna(family):
                val = find_mode(df[df['family'] == family]['local_name'])
            if val:
                changes.append({'species': sci, 'field': 'local_name', 'old': df.at[idx, 'local_name'], 'new': val})
                df.at[idx, 'local_name'] = val

        # max_size
        if 'max_size' in df.columns and pd.isna(df.at[idx, 'max_size']):
            val = None
            if genus:
                valnum = median_numeric(df[df['scientific_name'].str.startswith(genus + ' ', na=False)]['max_size'])
                if valnum:
                    val = f"{valnum} cm"
            if not val and pd.notna(family):
                valnum = median_numeric(df[df['family'] == family]['max_size'])
                if valnum:
                    val = f"{valnum} cm"
            if val:
                changes.append({'species': sci, 'field': 'max_size', 'old': df.at[idx, 'max_size'], 'new': val})
                df.at[idx, 'max_size'] = val

        # habitat
        if 'habitat' in df.columns and pd.isna(df.at[idx, 'habitat']):
            val = None
            if genus:
                val = find_mode(df[df['scientific_name'].str.startswith(genus + ' ', na=False)]['habitat'])
            if not val and pd.notna(family):
                val = find_mode(df[df['family'] == family]['habitat'])
            if val:
                changes.append({'species': sci, 'field': 'habitat', 'old': df.at[idx, 'habitat'], 'new': val})
                df.at[idx, 'habitat'] = val

    # Recalculate completeness_score
    target_cols = ['scientific_name', 'family', 'order', 'iucn_status', 'local_name', 'max_size', 'habitat']
    for c in target_cols:
        if c not in df.columns:
            df[c] = pd.NA
    df['completeness_score'] = df[target_cols].notna().sum(axis=1)

    # Save new master and report
    df.to_csv(MASTER_V3, index=False)
    pd.DataFrame(changes).to_csv(REPORT, index=False)
    return len(changes), REPORT

if __name__ == '__main__':
    cnt, rpt = enrich_top(N)
    print(f"✅ Enriched {cnt} fields for top {N} priorities. Report: {rpt}")