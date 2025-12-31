import os
from pypdf import PdfReader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PDF_DIR = os.path.join(ROOT, 'datasets')
OUT_DIR = os.path.join(ROOT, 'datasets', 'stockfish_metadata', 'pdf_texts')
os.makedirs(OUT_DIR, exist_ok=True)

pdf_files = []
for dirpath, _, filenames in os.walk(PDF_DIR):
    for fn in filenames:
        if fn.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(dirpath, fn))

print(f'Found {len(pdf_files)} PDF files')
for pdf in pdf_files:
    basename = os.path.basename(pdf)
    out_txt = os.path.join(OUT_DIR, basename + '.txt')
    try:
        reader = PdfReader(pdf)
        texts = []
        for page in reader.pages:
            texts.append(page.extract_text() or '')
        text = '\n\n'.join(texts)
        with open(out_txt, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f'Extracted: {basename} -> {out_txt}')
    except Exception as e:
        print(f'Failed to extract {basename}: {e}')
print('Done')
