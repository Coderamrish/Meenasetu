import csv
import os
import re
import json
from collections import defaultdict

BASE = r"c:\Users\AMRISH\Documents\Meenasetu\datasets\stockfish_metadata"
DRAFT_IN = os.path.join(BASE, 'fish_mapping_draft_enriched.csv')
FINAL_IN = os.path.join(BASE, 'fish_mapping.csv')
PDF_EX = os.path.join(BASE, 'pdf_extracted_parameters.csv')
RAG_DIR = os.path.join(BASE, 'rag_docs')
RAG_OUT = os.path.join(BASE, 'rag_index.jsonl')
SUMMARY_OUT = os.path.join(BASE, 'rag_index_summary.md')

EXP_KEYS = ['experiment','experimental','indoor','laboratory','lab','trial','exposed','reared at','study','in vitro','mortality','simulation','exposure','hatchery']
ICAR_KEYS = ['icar','cifa','icarcifa']

# load parameter rows to detect experimental vs field
param_rows = []
with open(PDF_EX, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        param_rows.append(row)

# group temps by species and mark experimental
temps_by_species = defaultdict(list)
for row in param_rows:
    sp = row['species_kw'].strip()
    if row['parameter'].lower().strip() != 'temperature':
        continue
    txt = (row.get('context','') + ' ' + row.get('value','')).lower()
    is_exp = any(k in txt for k in EXP_KEYS)
    is_icar = any(k in row.get('source','').lower() for k in ICAR_KEYS)
    temps_by_species[sp.lower()].append({'value': row['value'], 'context': row.get('context',''), 'source': row.get('source',''), 'experimental': is_exp, 'icar': is_icar})

# helper to extract numeric ranges
num_re = re.compile(r"(-?\d+(?:\.\d+)?)")
range_re = re.compile(r"(-?\d+(?:\.\d+)?)\s*[-–—]\s*(-?\d+(?:\.\d+)?)")

def parse_temp_val(v):
    if not v: return None
    m = range_re.search(v)
    if m:
        return float(m.group(1)), float(m.group(2))
    ms = num_re.findall(v)
    if ms:
        nums = [float(x) for x in ms]
        return min(nums), max(nums)
    return None

# read draft enriched
rows = []
with open(DRAFT_IN, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    fieldnames = r.fieldnames + ['temperature_note','preferred_source','experimental_temperature_included']
    for row in r:
        rows.append(row)

# refine per species
for row in rows:
    sp = row['fish_species']
    key = sp.lower()
    temps = temps_by_species.get(key, [])
    if not temps:
        # try alternate keys (handle 'Rohu' vs 'Labeo rohita' - check pdf species_kw normalized)
        alt = key
        temps = temps_by_species.get(alt, [])
    # decide experimental inclusion
    exp_only = all(t['experimental'] for t in temps) if temps else False
    any_icar = any(t['icar'] for t in temps)
    # calculate range preferring non-experimental if exists
    nonexp = [parse_temp_val(t['value']) for t in temps if not t['experimental']]
    nonexp = [p for p in nonexp if p]
    parsed = [parse_temp_val(t['value']) for t in temps if parse_temp_val(t['value'])]
    if nonexp:
        mins = [p[0] for p in nonexp]
        maxs = [p[1] for p in nonexp]
        mn = min(mins); mx = max(maxs)
        row['temperature_range'] = f"{mn} - {mx}°C" if mn!=mx else f"{mn}°C"
        row['temperature_note'] = 'Derived from non-experimental mentions when available.'
        row['experimental_temperature_included'] = 'no'
    elif parsed:
        mins = [p[0] for p in parsed]
        maxs = [p[1] for p in parsed]
        mn = min(mins); mx = max(maxs)
        row['temperature_range'] = f"{mn} - {mx}°C" if mn!=mx else f"{mn}°C"
        row['temperature_note'] = 'Only experimental/controlled conditions available — marked experimental in notes.'
        row['experimental_temperature_included'] = 'yes'
        if 'notes' in row and row['notes']:
            row['notes'] += ' Experimental values included.'
        else:
            row['notes'] = 'Experimental values included.'
    else:
        # no parsed numeric temps; leave as-is
        row['temperature_note'] = ''
        row['experimental_temperature_included'] = 'no'
    # preferred source
    if any_icar:
        row['preferred_source'] = 'ICAR (preferred)'
        row['notes'] = (row.get('notes','') + ' ICAR sources present — preferred for field recommendations.').strip()
        # bump confidence in final CSV when writing later
        row['notes'] = row['notes']
    else:
        row['preferred_source'] = ''

# write back enriched file
with open(DRAFT_IN, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)

# update final fish_mapping.csv confidence based on ICAR and experimental flag
final_rows = []
with open(FINAL_IN, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    cols = r.fieldnames
    for row in r:
        final_rows.append(row)

# map by fish_species
enriched_map = {r['fish_species']: r for r in rows}
for row in final_rows:
    sp = row['fish_species']
    enr = enriched_map.get(sp)
    if not enr: continue
    # set confidence rules
    conf = row.get('confidence','')
    if enr.get('preferred_source') and enr.get('preferred_source').lower().startswith('icar'):
        row['confidence'] = 'high'
    # mark notes
    note = enr.get('notes','')
    if not note:
        note = row.get('notes','')
    row['notes'] = note

with open(FINAL_IN, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in final_rows:
        w.writerow(r)

# Build RAG JSONL by chunking rag_docs
chunks = []
chunk_size = 800
overlap = 100
for fname in os.listdir(RAG_DIR):
    if not fname.lower().endswith('.txt'): continue
    path = os.path.join(RAG_DIR, fname)
    with open(path, encoding='utf-8') as f:
        txt = f.read()
    species = os.path.splitext(fname)[0].replace('_',' ')
    # simple chunk by characters
    start = 0
    i = 0
    while start < len(txt):
        end = min(start + chunk_size, len(txt))
        chunk = txt[start:end]
        meta = {'species': species, 'source_file': fname, 'chunk_index': i}
        chunks.append({'id': f"{species}_{i}", 'text': chunk, 'meta': meta})
        i += 1
        start = end - overlap

with open(RAG_OUT, 'w', encoding='utf-8') as f:
    for c in chunks:
        f.write(json.dumps(c, ensure_ascii=False) + '\n')

# write a summary
with open(SUMMARY_OUT, 'w', encoding='utf-8') as f:
    f.write('# RAG index summary\n\n')
    f.write(f'Total docs: {len(os.listdir(RAG_DIR))}\n')
    f.write(f'Total chunks: {len(chunks)}\n')

print('Refinement and RAG JSONL generation complete.')
print('Wrote:', DRAFT_IN, 'and updated', FINAL_IN, 'and', RAG_OUT)
