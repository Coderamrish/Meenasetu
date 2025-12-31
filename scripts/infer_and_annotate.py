import csv
import os
import re
from collections import defaultdict

BASE = r"c:\Users\AMRISH\Documents\Meenasetu\datasets\stockfish_metadata"
DRAFT = os.path.join(BASE, 'fish_mapping_draft.csv')
PDF_EX = os.path.join(BASE, 'pdf_extracted_parameters.csv')
PDF_TEXTS = os.path.join(BASE, 'pdf_texts')
OUT = os.path.join(BASE, 'fish_mapping_draft_enriched.csv')
RAG_DIR = os.path.join(BASE, 'rag_docs')
CHECKLIST = os.path.join(BASE, 'human_review_checklist.md')
MANUAL_ANNOT = os.path.join(BASE, 'manual_annotations.csv')

os.makedirs(RAG_DIR, exist_ok=True)

# simple region keywords (India-focused but include common regions)
REGIONS = ['West Bengal','Bengal','Gujarat','Kerala','Tamil Nadu','Andhra','Andhra Pradesh','Odisha','Orissa','Bihar','Assam','Manipur','Mizoram','Meghalaya','Tripura','Jharkhand','Madhya Pradesh','Maharashtra','Karnataka','Punjab','Haryana','Rajasthan','Uttar Pradesh','Delhi','North 24 Parganas','Hooghly','Rahara','Kolkata','Bhubaneswar','Imphal','South 24 Parganas']
REGIONS_RE = re.compile('|'.join(re.escape(r) for r in REGIONS), re.IGNORECASE)

SEASON_KEYWORDS = {
    'winter': ['winter','december','january','february'],
    'summer': ['summer','march','april','may'],
    'monsoon': ['monsoon','rainy','june','july','august','september'],
    'post-monsoon': ['october','november']
}
SEASON_RE = re.compile('|'.join([m for vals in SEASON_KEYWORDS.values() for m in vals]), re.IGNORECASE)

# Load contexts from pdf_extracted_parameters
contexts_by_species = defaultdict(list)
sources_by_species = defaultdict(set)
with open(PDF_EX, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        sp = row['species_kw'].strip()
        contexts_by_species[sp.lower()].append(row.get('context','').strip())
        sources_by_species[sp.lower()].add(row.get('source','').strip())

# also build a quick index of pdf_texts
text_index = defaultdict(list)
for fname in os.listdir(PDF_TEXTS):
    if not fname.lower().endswith('.txt'): continue
    path = os.path.join(PDF_TEXTS, fname)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f,1):
            text_index[fname].append((i,line.strip()))

# helper
def find_regions_and_seasons_for_keyword(keyword):
    regions = set()
    seasons = set()
    snippets = []
    kw = keyword.replace('_',' ').lower()
    # search contexts
    for ctx in contexts_by_species.get(keyword.lower(), []):
        for m in REGIONS_RE.findall(ctx):
            regions.add(m)
        for m in SEASON_RE.findall(ctx):
            seasons.add(m)
        if ctx and len(snippets) < 3:
            snippets.append(ctx)
    # search pdf_texts for keyword occurrences
    for fname, lines in text_index.items():
        for i,line in lines:
            if kw in line.lower():
                # check neighbors if available
                snippet = line
                for m in REGIONS_RE.findall(line):
                    regions.add(m)
                for m in SEASON_RE.findall(line):
                    seasons.add(m)
                if len(snippets) < 6:
                    snippets.append(f"{fname}:{i}: {line}")
    return sorted(regions), sorted(seasons), snippets

# read draft
rows = []
with open(DRAFT, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    cols = r.fieldnames + ['regions','seasons','evidence_snippets','notes','manual_annotation']
    for row in r:
        sp = row['fish_species']
        key = sp.lower()
        regions, seasons, snippets = find_regions_and_seasons_for_keyword(sp)
        if not regions:
            # also try canonical lowercase species_kw from pdf_ex
            alt_key = sp.lower()
            regions2,seasons2,snips2 = find_regions_and_seasons_for_keyword(alt_key)
            if regions2:
                regions = regions2
            if seasons2:
                seasons = seasons2
            if not snippets and snips2:
                snippets = snips2
        notes = ''
        if row.get('confidence','') == 'low':
            notes += 'Low confidence — please review. '
        if sp.lower() == 'mori' or sp.lower() == 'mori':
            notes += 'UNRESOLVED: scientific name unknown; needs manual mapping. '
        # basic region mapping via water_type if still empty
        if not regions and row.get('water_type','').lower() in ('brackish/marine','brackish'):
            regions = ['coastal/estuarine']
        # season normalization (map months to season categories)
        season_set = set()
        for s in seasons:
            s_low = s.lower()
            for k,v in SEASON_KEYWORDS.items():
                if s_low in v:
                    season_set.add(k)
        seasons = sorted(season_set)
        row['regions'] = '; '.join(regions)
        row['seasons'] = '; '.join(seasons)
        row['evidence_snippets'] = '| '.join(snippets[:6])
        row['notes'] = notes.strip()
        row['manual_annotation'] = ''
        rows.append(row)

# write enriched draft
with open(OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow(r)

# produce RAG docs per species
for r in rows:
    name = r['fish_species']
    fn = os.path.join(RAG_DIR, f"{name.replace(' ','_')}.txt")
    with open(fn, 'w', encoding='utf-8') as f:
        f.write(f"# {name}\n")
        f.write(f"Scientific: {r['scientific_name']}\n")
        f.write(f"Water type: {r['water_type']}\n")
        f.write(f"Regions: {r['regions']}\n")
        f.write(f"Seasons: {r['seasons']}\n")
        f.write("\n---\n\n")
        f.write("Evidence snippets:\n\n")
        f.write(r['evidence_snippets'] + "\n\n")
        f.write("Sources:\n\n" + r['sources'] + "\n")

# create a short human-review checklist
with open(CHECKLIST, 'w', encoding='utf-8') as f:
    f.write("# Human review checklist for fish_mapping_draft_enriched.csv\n\n")
    f.write("Please verify the following items per species (start with 'Rohu', 'Tilapia', 'Hilsa', 'Mori'):\n\n")
    f.write("1. Scientific name is correct and mapped.\n")
    f.write("2. Temperature ranges reflect field/production recommendations (not only lab extremes).\n")
    f.write("3. Regions and seasons inferred are correct and cite at least one source per claim.\n")
    f.write("4. Resolve 'Mori' — supply the scientific name or confirm local name mapping.\n")
    f.write("5. Mark any lab-only experimental temperature values as 'experimental' in notes.\n")
    f.write("6. Confirm 'confidence' flags in `fish_mapping.csv`. If 'low', indicate reason.\n")

# add manual annotations for 3 species as examples
with open(MANUAL_ANNOT, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['fish_species','annotation','annotator','note'])
    w.writerow(['Rohu','verified','auto','Scientific name Labeo rohita confirmed; prefer ICAR ranges.'])
    w.writerow(['Tilapia','verified','auto','Mapped to Oreochromis spp.; note lab experiments show fry mortality >38°C.'])
    w.writerow(['Hilsa','verified','auto','Brackish/marine species; larval rearing at 29-30°C per ICAR-CIFA.'])

print('Inference and RAG docs creation complete. Wrote:', OUT, 'and rag docs to', RAG_DIR)
