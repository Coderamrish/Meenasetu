import csv
import re
import json
from collections import defaultdict

IN = r"c:\Users\AMRISH\Documents\Meenasetu\datasets\stockfish_metadata\pdf_extracted_parameters.csv"
OUT_DRAFT = r"c:\Users\AMRISH\Documents\Meenasetu\datasets\stockfish_metadata\fish_mapping_draft.csv"
OUT_CSV = r"c:\Users\AMRISH\Documents\Meenasetu\datasets\stockfish_metadata\fish_mapping.csv"
OUT_JSON = r"c:\Users\AMRISH\Documents\Meenasetu\datasets\stockfish_metadata\fish_mapping.json"
OUT_SUMMARY = r"c:\Users\AMRISH\Documents\Meenasetu\datasets\stockfish_metadata\fish_mapping_summary.md"

# Canonical mapping (common name -> canonical common name and scientific)
CANON = {
    'rohu': ('Rohu','Labeo rohita','freshwater'),
    'labeo rohita': ('Rohu','Labeo rohita','freshwater'),
    'catla': ('Catla','Catla catla','freshwater'),
    'catla catla': ('Catla','Catla catla','freshwater'),
    'commoncarp': ('Common Carp','Cyprinus carpio','freshwater'),
    'common carp': ('Common Carp','Cyprinus carpio','freshwater'),
    'silvercarp': ('Silver Carp','Hypophthalmichthys molitrix','freshwater'),
    'silver carp': ('Silver Carp','Hypophthalmichthys molitrix','freshwater'),
    'tilapia': ('Tilapia','Oreochromis spp.','freshwater'),
    'hilsa': ('Hilsa','Tenualosa ilisha','brackish/marine'),
    'catfish': ('Catfish','Clarias spp. / Pangasius spp.','freshwater/brackish'),
    'clarias': ('Clarias','Clarias spp.','freshwater'),
    'grass carp': ('Grass Carp','Ctenopharyngodon idella','freshwater'),
    'mrigal': ('Mrigal','Cirrhinus mrigala','freshwater'),
    'labeo rohita': ('Rohu','Labeo rohita','freshwater')
}

# We'll keep unknown labels as-is but mark scientific as TBD

PARAMS = defaultdict(lambda: {'temperature_vals': [], 'growth_days': [], 'stocking_vals': [], 'raws': [], 'sources': set()})

num_re = re.compile(r"(-?\d+(?:\.\d+)?)")
range_re = re.compile(r"(-?\d+(?:\.\d+)?)\s*[-–—]\s*(-?\d+(?:\.\d+)?)")

def parse_temperature(val):
    v = val.replace('°','').replace('C','').replace('c','').strip()
    # try range
    m = range_re.search(val)
    if m:
        return float(m.group(1)), float(m.group(2))
    ms = num_re.findall(v)
    if ms:
        nums = [float(x) for x in ms]
        if len(nums) == 1:
            return nums[0], nums[0]
        return min(nums), max(nums)
    return None


def parse_duration(val):
    # returns days as int approx and a textual unit
    txt = val.lower()
    if 'year' in txt or 'yr' in txt:
        m = num_re.search(txt)
        if m: return int(float(m.group(1))*365)
    if 'month' in txt or 'mo' in txt:
        m = num_re.search(txt)
        if m: return int(float(m.group(1))*30)
    if 'day' in txt or 'days' in txt or 'd' in txt:
        m = num_re.search(txt)
        if m: return int(float(m.group(1)))
    # fallback: single number likely days
    m = num_re.search(txt)
    if m:
        return int(float(m.group(1)))
    return None


def summarize_range(pairs):
    if not pairs:
        return ''
    mins = [p[0] for p in pairs]
    maxs = [p[1] for p in pairs]
    mn = min(mins)
    mx = max(maxs)
    if mn == mx:
        return f"{mn}°C"
    return f"{mn} - {mx}°C"


def days_to_readable(d):
    if d is None: return ''
    if d >= 365:
        y = d/365
        return f"{round(y,2)} yr ({d} days)"
    if d >= 30:
        m = d/30
        return f"{round(m,1)} mo ({d} days)"
    return f"{d} days"

# read input
with open(IN, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        kw = row['species_kw'].strip()
        param = row['parameter'].strip().lower()
        val = row['value'].strip()
        src = row['source'].strip()
        key = kw.lower()
        canonical = CANON.get(key, (kw, 'TBD', 'unknown'))
        name = canonical[0]
        entry = PARAMS[name]
        entry['raws'].append((param, val, row.get('context','')))
        entry['sources'].add(src)
        if param == 'temperature':
            parsed = parse_temperature(val)
            if parsed:
                entry['temperature_vals'].append(parsed)
        elif 'growth' in param:
            d = parse_duration(val)
            if d is not None:
                entry['growth_days'].append(d)
        elif 'stock' in param:
            # store raw for now
            entry['stocking_vals'].append(val)

# create draft and final outputs
with open(OUT_DRAFT, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['fish_species','scientific_name','water_type','temperature_range','growth_duration_summary','stocking_density_summary','sources','evidence_count','notes'])
    for sp, data in sorted(PARAMS.items()):
        key = sp.lower()
        sci = CANON.get(key, (sp,'TBD','unknown'))[1]
        water = CANON.get(key, (sp,'TBD','unknown'))[2]
        temp = summarize_range(data['temperature_vals'])
        if not temp:
            temp = ''
        gd_min = min(data['growth_days']) if data['growth_days'] else None
        gd_max = max(data['growth_days']) if data['growth_days'] else None
        if gd_min is None:
            gd_summary = ''
        elif gd_min == gd_max:
            gd_summary = days_to_readable(gd_min)
        else:
            gd_summary = f"{days_to_readable(gd_min)} - {days_to_readable(gd_max)}"
        stock = '; '.join(sorted(set(data['stocking_vals']))) if data['stocking_vals'] else ''
        sources = '; '.join(sorted(data['sources']))
        notes = f"{len(data['raws'])} raw matches"
        w.writerow([sp, sci, water, temp, gd_summary, stock, sources, len(data['raws']), notes])

# write final CSV/JSON with same rows but add confidence heuristic
out_rows = []
with open(OUT_DRAFT, newline='', encoding='utf-8') as f_in:
    r = csv.DictReader(f_in)
    for row in r:
        # simple confidence: >=2 distinct sources and at least one numeric param
        srcs = [s for s in row['sources'].split(';') if s]
        has_param = bool(row['temperature_range'] or row['growth_duration_summary'] or row['stocking_density_summary'])
        confidence = 'high' if (len(srcs) >= 2 and has_param) else ('medium' if (len(srcs) >=1 and has_param) else 'low')
        row['confidence'] = confidence
        out_rows.append(row)

with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
    cols = ['fish_species','scientific_name','water_type','temperature_range','growth_duration_summary','stocking_density_summary','sources','evidence_count','confidence','notes']
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in out_rows:
        w.writerow({k: r.get(k,'') for k in cols})

with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(out_rows, f, indent=2, ensure_ascii=False)

with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
    f.write("# Fish mapping summary\n\n")
    f.write("Method: extracted parameters from local ICAR/Annual report PDFs and consolidated into per-species ranges. Temperature ranges use numeric min/max from mentions; growth durations are normalized to days. Sources column lists the originating text files.\n\n")
    f.write("Files generated:\n\n- datasets/stockfish_metadata/fish_mapping_draft.csv (detailed evidence summary)\n- datasets/stockfish_metadata/fish_mapping.csv (final aggregated table with confidence)\n- datasets/stockfish_metadata/fish_mapping.json (JSON export)\n")

print('Aggregation complete. Draft and final files written.')
