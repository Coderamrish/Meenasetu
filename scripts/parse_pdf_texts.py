import os
import re
import csv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TXT_DIR = os.path.join(ROOT, 'datasets', 'stockfish_metadata', 'pdf_texts')
OUT_CSV = os.path.join(ROOT, 'datasets', 'stockfish_metadata', 'pdf_extracted_parameters.csv')

species_keywords = [
    'Rohu', 'Labeo rohita',
    'Catla', 'Catla catla',
    'Common carp', 'Cyprinus carpio',
    'Silver carp', 'Hypophthalmichthys molitrix',
    'Mori', 'Tilapia', 'Oreochromis', 'Catfish', 'Clarias', 'Murrel', 'Channa',
    'Mrigal', 'Grass carp', 'Hilsa', 'Tilapia nilotica', 'Tilapia mossambica',
    'Ornamental', 'Koi', 'Goldfish'
]

# regex patterns
temp_re = re.compile(r"(\d{1,2}(?:\.\d)?\s*(?:°[Cc]|degC|C)?)\s*(?:[-–toto]{1,3}\s*(\d{1,2}(?:\.\d)?\s*(?:°[Cc]|degC|C)?))?", re.IGNORECASE)
stock_re = re.compile(r"(stock(?:ing)?\s*density|stocking)[:\s]*([0-9]+(?:\.[0-9]+)?\s*(?:nos|numbers|fish|kg)?/?(?:ha|hectare|ha)?)", re.IGNORECASE)
duration_re = re.compile(r"(\d{1,2}(?:\.\d)?\s*(?:months|month|days|yrs|years))", re.IGNORECASE)

findings = []

for fn in os.listdir(TXT_DIR):
    if not fn.lower().endswith('.txt'):
        continue
    path = os.path.join(TXT_DIR, fn)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        for kw in species_keywords:
            if kw.lower() in line.lower():
                context = ' '.join(lines[max(0,i-2):min(len(lines),i+3)])
                # search temp (require context keywords and reasonable ranges)
                temp_matches = temp_re.findall(context)
                stock_matches = stock_re.findall(context)
                dur_matches = duration_re.findall(context)
                # filter temperature matches: keep only if context mentions 'temp' or '°C' or 'temperature' or 'optimum'
                good_temp_matches = []
                for t in temp_matches:
                    txt = ' '.join(t).lower()
                    if any(k in context.lower() for k in ['temp', '°c', 'temperature', 'optimum', 'water temperature']):
                        # extract numeric part
                        num_match = re.search(r"(\d{1,2}(?:\.\d)?)", t[0])
                        if num_match:
                            valnum = float(num_match.group(1))
                            if 5 <= valnum <= 40:  # reasonable temp range
                                val = t[0]
                                if t[1]:
                                    val = val + ' - ' + t[1]
                                good_temp_matches.append(val.strip())
                # filter duration matches: require 'month' or 'grow' or 'culture'
                good_dur_matches = []
                for d in dur_matches:
                    if any(k in context.lower() for k in ['month', 'grow', 'culture', 'grow-out', 'grow out']):
                        good_dur_matches.append(d.strip())
                # filter stock matches: ensure "stock":
                good_stock_matches = []
                for s in stock_matches:
                    if 'stock' in s[0].lower() or 'stock' in context.lower():
                        good_stock_matches.append(s[1].strip() if len(s)>1 else s[0])
                if good_temp_matches or good_stock_matches or good_dur_matches:
                    for val in good_temp_matches:
                        findings.append({'species_kw':kw, 'parameter':'temperature', 'value':val, 'source':fn, 'context':context[:400]})
                    for val in good_stock_matches:
                        findings.append({'species_kw':kw, 'parameter':'stocking_density', 'value':val, 'source':fn, 'context':context[:400]})
                    for val in good_dur_matches:
                        findings.append({'species_kw':kw, 'parameter':'growth_duration', 'value':val, 'source':fn, 'context':context[:400]})

# write CSV
with open(OUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['species_kw','parameter','value','source','context'])
    writer.writeheader()
    for r in findings:
        writer.writerow(r)

print(f'Found {len(findings)} parameter matches across PDF texts. Output: {OUT_CSV}')
