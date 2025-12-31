import pandas as pd
import os
from pathlib import Path
import requests
import json
import time
import difflib


def _gbif_lookup_species(name):
    """Lookup species via GBIF species/match API and return (family, order) or (None, None).
    Returns quickly on network error or timeout."""
    if not name or pd.isna(name):
        return (None, None)
    q = {"name": name}
    try:
        r = requests.get("https://api.gbif.org/v1/species/match", params=q, timeout=5)
        r.raise_for_status()
        data = r.json()
        fam = data.get("family")
        order_ = data.get("order")
        return (fam, order_)
    except requests.exceptions.RequestException:
        return (None, None)
    except Exception:
        return (None, None)

# 1. Define Paths

def _find_master():
    # Prefer any variant like fish_mapping_master_v2.csv, then fallback to base filename
    candidates = sorted(Path("data/final").glob("fish_mapping_master*.csv"), key=lambda p: p.name, reverse=True)
    if candidates:
        return candidates[0]
    return Path("data/final/fish_mapping_master.csv")

MASTER = _find_master()
TAXONOMY = Path("data/reference/family_order_map.csv")
GENUS_MAP_PATH = Path("data/reference/genus_family_map.csv")
OUTPUT = Path("data/final/fish_mapping_master_v2.csv")

def fix_fish_data():
    # 2. Check if master file exists
    if not MASTER.exists():
        print(f"❌ Error: {MASTER} not found!")
        return

    # 3. Load Master Data FIRST (treat common placeholders as NA)
    df = pd.read_csv(MASTER, na_values=["", "NA", "None", "nan", "NaN"]) 
    filled_families = 0
    filled_orders = 0

    # 4. Step A: Infer Family from Genus
    if GENUS_MAP_PATH.exists():
        genus_map = pd.read_csv(GENUS_MAP_PATH, na_values=["", "NA", "None", "nan", "NaN"]) 
        # Map genera case-insensitively
        genus_dict = {str(g).strip().lower(): f for g, f in zip(genus_map['genus'], genus_map['family']) if pd.notna(g)}

        # Augment genus map with observed families in master (case-insensitive)
        for _, r in df[df['family'].notna()].iterrows():
            sci = r.get('scientific_name')
            fam = r.get('family')
            if pd.isna(sci) or pd.isna(fam):
                continue
            g = str(sci).split()[0].strip().lower()
            if g and g not in genus_dict:
                genus_dict[g] = fam

        def infer_family(row):
            nonlocal filled_families
            if pd.notna(row.get("family")):
                return row.get("family")

            sci = row.get("scientific_name")
            if pd.isna(sci):
                return pd.NA

            genus = str(sci).split()[0].strip().lower()
            if genus in genus_dict:
                filled_families += 1
                return genus_dict[genus]
            return pd.NA

        df["family"] = df.apply(infer_family, axis=1)

        # 4b. Fuzzy genus matching to handle typos/variants (use difflib)
        try:
            missing_family_idx = df[df['family'].isna() & df['scientific_name'].notna()].index.tolist()
            genus_keys = list(genus_dict.keys())
            filled_fuzzy = 0
            for idx in missing_family_idx:
                sci = df.at[idx, 'scientific_name']
                genus = str(sci).split()[0].strip().lower()
                matches = difflib.get_close_matches(genus, genus_keys, n=1, cutoff=0.8)
                if matches:
                    df.at[idx, 'family'] = genus_dict[matches[0]]
                    filled_fuzzy += 1
            if filled_fuzzy:
                filled_families += filled_fuzzy
                print(f"🔧 Fuzzy genus matching filled {filled_fuzzy} family values")
        except Exception:
            pass

    # 5. Step B: Infer Order from Family
    if TAXONOMY.exists():
        taxonomy = pd.read_csv(TAXONOMY, na_values=["", "NA", "None", "nan", "NaN"]) 
        # map family -> order using lower-cased keys to be robust
        family_to_order = {str(f).strip().lower(): o for f, o in zip(taxonomy["family"], taxonomy["order"]) if pd.notna(f)}

        for idx, row in df.iterrows():
            family = row.get("family")
            order = row.get("order")

            if pd.notna(family) and pd.isna(order):
                fam_key = str(family).strip().lower()
                if fam_key in family_to_order:
                    df.at[idx, "order"] = family_to_order[fam_key]
                    filled_orders += 1

    # Optional Step D: External lookup (GBIF) for missing family/order
    def _gbif_lookup_species(name):
        """Lookup species via GBIF species/match API and return (family, order) or (None, None)"""
        if not name or pd.isna(name):
            return (None, None)
        q = {"name": name}
        try:
            r = requests.get("https://api.gbif.org/v1/species/match", params=q, timeout=10)
            if r.status_code == 200:
                data = r.json()
                fam = data.get("family")
                order_ = data.get("order")
                return (fam, order_)
        except Exception:
            pass
        return (None, None)
    
    # 6. Step C: Update Completeness Score
    # Recalculate how many fields are filled for each row (out of your target columns)
    target_cols = ['scientific_name', 'family', 'order', 'iucn_status', 'local_name', 'max_size', 'habitat']
    for c in target_cols:
        if c not in df.columns:
            df[c] = pd.NA
    df['completeness_score'] = df[target_cols].notna().sum(axis=1)

    # 7. Save Results
    OUTPUT.parent.mkdir(parents=True, exist_ok=True) # Create dir if missing
    df.to_csv(OUTPUT, index=False)

    print(f"==========================================")
    print(f"✅ FAMILY INFERRED: {filled_families}")
    print(f"✅ ORDERS FILLED:   {filled_orders}")
    print(f"📄 SAVED TO:        {OUTPUT}")
    print(f"==========================================")


def run_fix(external=False, cache_path=Path('data/final/taxonomy_cache.json'), max_queries=200):
    """Wrap the fix and optionally perform external lookups to fill remaining gaps."""
    # Load existing cache
    cache = {}
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    # Read the just-saved output to operate on latest changes
    df = pd.read_csv(OUTPUT, na_values=['', 'NA', 'None', 'nan', 'NaN'])
    added = 0

    if external:
        # quick check for network/connectivity to GBIF
        try:
            resp = requests.get('https://api.gbif.org', timeout=2)
            if resp.status_code >= 400:
                raise requests.exceptions.RequestException('GBIF not reachable')
            network_ok = True
        except Exception:
            print('⚠️  External lookup requested but network/GBIF not reachable; skipping external queries')
            network_ok = False

        if not network_ok:
            missing_idx = []
        else:
            missing_idx = df[df['family'].isna()].index.tolist()

        total_to_query = min(len(missing_idx), max_queries)
        print(f"🔎 External lookup enabled: {len(missing_idx)} species missing family; querying up to {total_to_query}")
        queried = 0
        for i, idx in enumerate(missing_idx, 1):
            if queried >= total_to_query:
                break
            sci = df.at[idx, 'scientific_name']
            if not sci or pd.isna(sci):
                continue
            if sci in cache:
                fam, ord_ = cache[sci]
            else:
                fam, ord_ = _gbif_lookup_species(sci)
                cache[sci] = (fam, ord_)
                queried += 1
                # be polite with API
                time.sleep(0.2)

            if fam and pd.isna(df.at[idx, 'family']):
                df.at[idx, 'family'] = fam
                added += 1
            if ord_ and pd.isna(df.at[idx, 'order']):
                df.at[idx, 'order'] = ord_

    # Recalculate completeness
    target_cols = ['scientific_name', 'family', 'order', 'iucn_status', 'local_name', 'max_size', 'habitat']
    for c in target_cols:
        if c not in df.columns:
            df[c] = pd.NA
    df['completeness_score'] = df[target_cols].notna().sum(axis=1)

    # Save updated master
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    # Save cache
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    print(f"🔁 External lookup filled {added} family values and updated {OUTPUT}")
    return added


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fix taxonomy and optionally query GBIF to fill families/orders')
    parser.add_argument('--external', action='store_true', help='Enable external GBIF lookups')
    parser.add_argument('--cache', help='Path to cache JSON file', default='data/final/taxonomy_cache.json')
    parser.add_argument('--max-queries', type=int, default=200, help='Maximum number of external queries to run')

    args = parser.parse_args()

    fix_fish_data()
    if args.external:
        run_fix(external=True, cache_path=Path(args.cache), max_queries=args.max_queries)

