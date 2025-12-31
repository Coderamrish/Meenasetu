"""Interactive helper to add rows to the master fish mapping CSV.

Features:
- Chooses the best master file automatically (v3 -> v2 -> master.csv -> fish_mapping.csv)
- Interactive prompt to add one or more rows following the CSV header
- Safe backup before writing
- Recalculates `completeness_score` and saves new file as same master
- Optionally runs `scripts/fish_data_validator.py` after changes
- Batch import from a CSV is also supported (merge without duplicates)

Usage:
- Interactive: python scripts/add_fish_entry.py
- Batch import: python scripts/add_fish_entry.py --batch new_rows.csv
- Specify master explicitly: python scripts/add_fish_entry.py --master data/final/fish_mapping_master_v3.csv
"""

import argparse
import pandas as pd
from pathlib import Path
import shutil
import subprocess
import datetime
import sys
import json

PREFERRED = ["data/final/fish_mapping_master_v3.csv",
             "data/final/fish_mapping_master_v2.csv",
             "data/final/fish_mapping_master.csv",
             "data/final/fish_mapping.csv"]


def choose_master(explicit=None):
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        else:
            raise FileNotFoundError(f"Master file not found: {p}")
    for p in PREFERRED:
        p = Path(p)
        if p.exists():
            return p
    raise FileNotFoundError("No master file found. Create data/final/fish_mapping.csv first.")


def backup_file(p: Path):
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    dst = p.with_suffix(p.suffix + f'.bak.{stamp}')
    shutil.copy2(p, dst)
    return dst


def recalc_completeness(df: pd.DataFrame):
    target_cols = ['scientific_name', 'family', 'order', 'iucn_status', 'local_name', 'max_size', 'habitat']
    for c in target_cols:
        if c not in df.columns:
            df[c] = pd.NA
    df['completeness_score'] = df[target_cols].notna().sum(axis=1)
    return df


def interactive_add(master: Path):
    df = pd.read_csv(master, na_values=['', 'NA', 'None', 'nan', 'NaN'])
    cols = df.columns.tolist()
    print(f"Adding rows to: {master}")
    print("Header columns:")
    print(", ".join(cols))
    added = 0
    rows = []

    while True:
        print('\nEnter new row values (press Enter to leave blank)')
        row = {}
        for c in cols:
            val = input(f" {c}: ")
            if val.strip() == "":
                row[c] = pd.NA
            else:
                row[c] = val.strip()
        rows.append(row)
        added += 1
        cont = input('Add another row? (y/N): ').strip().lower()
        if cont != 'y':
            break

    if not rows:
        print('No rows entered. Exiting.')
        return 0

    # Backup and append
    bak = backup_file(master)
    print(f"Backup saved to {bak}")

    new_df = pd.DataFrame(rows)
    combined = pd.concat([df, new_df], ignore_index=True)
    combined = recalc_completeness(combined)
    combined.to_csv(master, index=False, encoding='utf-8')
    print(f"Appended {added} rows to {master}")
    return added


def batch_import(master: Path, batch_csv: Path):
    if not batch_csv.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_csv}")
    df = pd.read_csv(master, na_values=['', 'NA', 'None', 'nan', 'NaN'])
    inc = pd.read_csv(batch_csv, na_values=['', 'NA', 'None', 'nan', 'NaN'])
    # Ensure same columns
    for c in inc.columns:
        if c not in df.columns:
            df[c] = pd.NA
    merged = pd.concat([df, inc], ignore_index=True)
    # Drop duplicates by scientific_name if present
    if 'scientific_name' in merged.columns:
        merged = merged.drop_duplicates(subset=['scientific_name'], keep='first')

    bak = backup_file(master)
    print(f"Backup saved to {bak}")
    merged = recalc_completeness(merged)
    merged.to_csv(master, index=False, encoding='utf-8')
    added = len(merged) - len(df)
    print(f"Imported {len(inc)} rows, net added {added} rows to {master}")
    return added


def maybe_run_validator(master: Path):
    run = input('Run validator now? (Y/n): ').strip().lower()
    if run == 'n':
        return
    cmd = [sys.executable, 'scripts/fish_data_validator.py', '--master', str(master)]
    print('Running validator... this may take a moment')
    subprocess.run(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add rows to fish master file interactively, via batch CSV, or via CLI JSON')
    parser.add_argument('--batch', help='Path to CSV file with rows to import')
    parser.add_argument('--master', help='Override master file path')
    parser.add_argument('--row-json', help='Add a single row via JSON string or path to JSON file')
    parser.add_argument('--auto-run', action='store_true', help='Automatically run validator after changes without prompting')
    args = parser.parse_args()

    master = choose_master(args.master)
    if args.batch:
        added = batch_import(master, Path(args.batch))
        print(f"Done. Added {added} rows.")
        if args.auto_run:
            subprocess.run([sys.executable, 'scripts/fish_data_validator.py', '--master', str(master)])
        else:
            maybe_run_validator(master)
    elif args.row_json:
        # Accept either a JSON string or a path to a JSON file
        j = args.row_json
        try:
            if Path(j).exists():
                with open(j, 'r', encoding='utf-8') as fj:
                    row = json.load(fj)
            else:
                row = json.loads(j)
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            raise SystemExit(1)

        # Append row
        df = pd.read_csv(master, na_values=['', 'NA', 'None', 'nan', 'NaN'])
        # Ensure all columns exist
        for c in df.columns:
            if c not in row:
                row[c] = None
        new_df = pd.DataFrame([row])
        bak = backup_file(master)
        print(f"Backup saved to {bak}")
        combined = pd.concat([df, new_df], ignore_index=True)
        combined = recalc_completeness(combined)
        combined.to_csv(master, index=False, encoding='utf-8')
        print(f"Added 1 row to {master}")
        if args.auto_run:
            subprocess.run([sys.executable, 'scripts/fish_data_validator.py', '--master', str(master)])
        else:
            maybe_run_validator(master)
    else:
        added = interactive_add(master)
        print(f"Done. Added {added} rows.")
        if args.auto_run:
            subprocess.run([sys.executable, 'scripts/fish_data_validator.py', '--master', str(master)])
        else:
            maybe_run_validator(master)