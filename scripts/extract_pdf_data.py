"""
MeenaSetu - PDF Data Extraction Pipeline
Extract fish species data from PDFs and create structured dataset
"""

import PyPDF2
import pdfplumber
import pandas as pd
import re
from pathlib import Path
import json
from typing import List, Dict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFDataExtractor:
    """Extract structured data from fish-related PDFs"""
    
    def __init__(self, pdf_dir: str = "datasets", output_dir: str = "datasets/processed"):
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Fish name patterns for Indian species
        self.fish_patterns = {
            'common_names': [
                'rohu', 'catla', 'mrigal', 'tilapia', 'pangasius',
                'magur', 'singhi', 'channa', 'murrel', 'carp',
                'hilsa', 'prawn', 'shrimp', 'goldfish', 'guppy'
            ],
            'scientific_prefixes': ['Labeo', 'Catla', 'Cirrhinus', 'Oreochromis',
                                  'Pangasianodon', 'Clarias', 'Heteropneustes',
                                  'Channa', 'Cyprinus', 'Hypophthalmichthys']
        }
    
    def extract_text_pypdf(self, pdf_path: Path) -> str:
        """Extract text using PyPDF2"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() + '\n'
            return text
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed for {pdf_path.name}: {e}")
            return ""
    
    def extract_tables_pdfplumber(self, pdf_path: Path) -> List[pd.DataFrame]:
        """Extract tables using pdfplumber"""
        tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        if table:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            df['source_page'] = page_num + 1
                            tables.append(df)
        except Exception as e:
            logger.error(f"pdfplumber extraction failed for {pdf_path.name}: {e}")
        
        return tables
    
    def find_fish_mentions(self, text: str) -> List[Dict]:
        """Find fish species mentions in text"""
        mentions = []
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines):
            line_lower = line.lower()
            
            # Check for common names
            for fish_name in self.fish_patterns['common_names']:
                if fish_name in line_lower:
                    mentions.append({
                        'line_number': line_num,
                        'line_text': line.strip(),
                        'fish_name': fish_name,
                        'type': 'common_name'
                    })
            
            # Check for scientific names
            for prefix in self.fish_patterns['scientific_prefixes']:
                if prefix in line:
                    mentions.append({
                        'line_number': line_num,
                        'line_text': line.strip(),
                        'fish_name': prefix,
                        'type': 'scientific_name'
                    })
        
        return mentions
    
    def extract_temperature_ranges(self, text: str) -> List[Dict]:
        """Extract temperature ranges from text"""
        # Patterns: 25-30°C, 25-30 °C, 25-30C, 25 to 30°C
        temp_pattern = r'(\d{1,2})\s*[-to]+\s*(\d{1,2})\s*[°]?\s*[Cc]'
        matches = re.finditer(temp_pattern, text)
        
        temperature_data = []
        for match in matches:
            temp_min = int(match.group(1))
            temp_max = int(match.group(2))
            
            # Get context (50 chars before and after)
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]
            
            temperature_data.append({
                'temp_min': temp_min,
                'temp_max': temp_max,
                'temp_avg': (temp_min + temp_max) / 2,
                'context': context.strip()
            })
        
        return temperature_data
    
    def extract_stocking_density(self, text: str) -> List[Dict]:
        """Extract stocking density information"""
        # Patterns: 5000/ha, 5000 per hectare, 5,000/ha
        density_pattern = r'(\d{1,2}[,.]?\d{0,3})\s*[/per]+\s*(?:ha|hectare)'
        matches = re.finditer(density_pattern, text, re.IGNORECASE)
        
        densities = []
        for match in matches:
            density = match.group(1).replace(',', '')
            
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end]
            
            densities.append({
                'stocking_density': int(density),
                'context': context.strip()
            })
        
        return densities
    
    def process_single_pdf(self, pdf_path: Path) -> Dict:
        """Process a single PDF and extract all data"""
        logger.info(f"Processing: {pdf_path.name}")
        
        # Extract text
        text = self.extract_text_pypdf(pdf_path)
        
        # Extract tables
        tables = self.extract_tables_pdfplumber(pdf_path)
        
        # Find patterns
        fish_mentions = self.find_fish_mentions(text)
        temperatures = self.extract_temperature_ranges(text)
        densities = self.extract_stocking_density(text)
        
        # Save extracted text
        text_output = self.output_dir / f"{pdf_path.stem}_text.txt"
        with open(text_output, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Save tables as CSV
        for i, table in enumerate(tables):
            table_output = self.output_dir / f"{pdf_path.stem}_table_{i}.csv"
            table.to_csv(table_output, index=False)
        
        return {
            'filename': pdf_path.name,
            'text_length': len(text),
            'num_tables': len(tables),
            'fish_mentions': len(fish_mentions),
            'temperature_ranges': len(temperatures),
            'stocking_densities': len(densities),
            'fish_data': fish_mentions[:10],  # Sample
            'temp_data': temperatures[:10],
            'density_data': densities[:10]
        }
    
    def process_all_pdfs(self) -> pd.DataFrame:
        """Process all PDFs in the directory"""
        logger.info(f"Scanning directory: {self.pdf_dir}")
        
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        results = []
        for pdf_path in pdf_files:
            try:
                result = self.process_single_pdf(pdf_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(results)
        summary_path = self.output_dir / "extraction_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        logger.info(f"Summary saved to: {summary_path}")
        return summary_df


class DataCleaner:
    """Clean and structure extracted data"""
    
    def __init__(self, processed_dir: str = "datasets/processed"):
        self.processed_dir = Path(processed_dir)
    
    def load_extracted_texts(self) -> Dict[str, str]:
        """Load all extracted text files"""
        texts = {}
        for text_file in self.processed_dir.glob("*_text.txt"):
            with open(text_file, 'r', encoding='utf-8') as f:
                texts[text_file.stem] = f.read()
        return texts
    
    def load_extracted_tables(self) -> List[pd.DataFrame]:
        """Load all extracted tables"""
        tables = []
        for csv_file in self.processed_dir.glob("*_table_*.csv"):
            try:
                df = pd.read_csv(csv_file)
                df['source_file'] = csv_file.stem
                tables.append(df)
            except Exception as e:
                logger.error(f"Failed to load {csv_file.name}: {e}")
        return tables
    
    def create_fish_species_list(self, texts: Dict[str, str]) -> pd.DataFrame:
        """Create list of unique fish species from texts"""
        species_data = []
        
        common_species = {
            'Rohu': 'Labeo rohita',
            'Catla': 'Catla catla',
            'Mrigal': 'Cirrhinus mrigala',
            'Silver Carp': 'Hypophthalmichthys molitrix',
            'Grass Carp': 'Ctenopharyngodon idella',
            'Common Carp': 'Cyprinus carpio',
            'Tilapia': 'Oreochromis niloticus',
            'Pangasius': 'Pangasianodon hypophthalmus',
            'Magur': 'Clarias batrachus',
            'Singhi': 'Heteropneustes fossilis',
            'Channa/Murrel': 'Channa striata'
        }
        
        for common, scientific in common_species.items():
            species_data.append({
                'common_name': common,
                'scientific_name': scientific,
                'category': self._categorize_species(common),
                'found_in_docs': sum(1 for text in texts.values() if common.lower() in text.lower())
            })
        
        df = pd.DataFrame(species_data)
        return df.sort_values('found_in_docs', ascending=False)
    
    def _categorize_species(self, fish_name: str) -> str:
        """Categorize fish species"""
        categories = {
            'major_carp': ['Rohu', 'Catla', 'Mrigal'],
            'exotic_carp': ['Silver Carp', 'Grass Carp', 'Common Carp', 'Tilapia'],
            'catfish': ['Pangasius', 'Magur', 'Singhi'],
            'murrel': ['Channa', 'Murrel']
        }
        
        for category, species_list in categories.items():
            if any(species in fish_name for species in species_list):
                return category
        
        return 'other'


class DatasetBuilder:
    """Build final fish_mapping.csv dataset"""
    
    def __init__(self, output_path: str = "data/final/fish_mapping.csv"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def create_template_dataset(self) -> pd.DataFrame:
        """Create template dataset with known species"""
        
        # Template data for Indian Major Carps
        data = {
            'common_name': [
                'Rohu', 'Catla', 'Mrigal',
                'Silver Carp', 'Grass Carp', 'Common Carp',
                'Tilapia', 'Pangasius',
                'Magur', 'Singhi',
                'Channa/Murrel'
            ],
            'scientific_name': [
                'Labeo rohita', 'Catla catla', 'Cirrhinus mrigala',
                'Hypophthalmichthys molitrix', 'Ctenopharyngodon idella', 'Cyprinus carpio',
                'Oreochromis niloticus', 'Pangasianodon hypophthalmus',
                'Clarias batrachus', 'Heteropneustes fossilis',
                'Channa striata'
            ],
            'water_type': [
                'pond', 'pond', 'pond',
                'pond', 'pond', 'pond',
                'pond', 'pond',
                'pond', 'pond',
                'pond'
            ],
            'region': [
                'West Bengal', 'West Bengal', 'West Bengal',
                'West Bengal', 'West Bengal', 'West Bengal',
                'Pan India', 'West Bengal',
                'West Bengal', 'West Bengal',
                'West Bengal'
            ],
            'season': [
                'all_year', 'all_year', 'all_year',
                'all_year', 'all_year', 'all_year',
                'all_year', 'all_year',
                'monsoon', 'monsoon',
                'all_year'
            ],
            'temp_min': [22, 22, 22, 20, 20, 15, 22, 22, 22, 22, 20],
            'temp_max': [32, 32, 32, 30, 30, 28, 35, 32, 32, 32, 30],
            'ph_min': [7.0, 7.0, 7.0, 7.0, 7.0, 6.5, 6.5, 7.0, 6.5, 6.5, 6.5],
            'ph_max': [8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.0],
            'stocking_density': [5000, 4000, 4000, 3000, 2000, 3000, 8000, 5000, 3000, 3000, 2000],
            'growth_duration_months': [12, 10, 12, 10, 12, 10, 6, 10, 10, 10, 12],
            'category': [
                'major_carp', 'major_carp', 'major_carp',
                'exotic_carp', 'exotic_carp', 'exotic_carp',
                'exotic', 'catfish',
                'catfish', 'catfish',
                'murrel'
            ],
            'feeding_habit': [
                'omnivore', 'planktivore', 'bottom_feeder',
                'planktivore', 'herbivore', 'omnivore',
                'omnivore', 'omnivore',
                'carnivore', 'carnivore',
                'carnivore'
            ],
            'market_value': [
                'high', 'high', 'high',
                'medium', 'medium', 'medium',
                'medium', 'medium',
                'high', 'high',
                'high'
            ]
        }
        
        df = pd.DataFrame(data)
        
        # Add derived features
        df['temp_avg'] = (df['temp_min'] + df['temp_max']) / 2
        df['ph_avg'] = (df['ph_min'] + df['ph_max']) / 2
        
        return df
    
    def save_dataset(self, df: pd.DataFrame):
        """Save dataset in multiple formats"""
        # CSV
        df.to_csv(self.output_path, index=False)
        logger.info(f"Saved CSV: {self.output_path}")
        
        # JSON
        json_path = self.output_path.with_suffix('.json')
        df.to_json(json_path, orient='records', indent=2)
        logger.info(f"Saved JSON: {json_path}")
        
        # Excel (optional)
        try:
            excel_path = self.output_path.with_suffix('.xlsx')
            df.to_excel(excel_path, index=False, sheet_name='Fish Mapping')
            logger.info(f"Saved Excel: {excel_path}")
        except Exception as e:
            logger.warning(f"Could not save Excel: {e}")


def main():
    """Main execution pipeline"""
    print("🐟 MeenaSetu - PDF Data Extraction Pipeline")
    print("=" * 60)
    
    # Step 1: Extract data from PDFs
    print("\n📄 Step 1: Extracting data from PDFs...")
    extractor = PDFDataExtractor()
    summary = extractor.process_all_pdfs()
    
    print("\n📊 Extraction Summary:")
    print(summary[['filename', 'text_length', 'num_tables', 'fish_mentions']])
    
    # Step 2: Clean and structure data
    print("\n🧹 Step 2: Cleaning and structuring data...")
    cleaner = DataCleaner()
    texts = cleaner.load_extracted_texts()
    tables = cleaner.load_extracted_tables()
    
    species_list = cleaner.create_fish_species_list(texts)
    print(f"\nFound {len(species_list)} unique species")
    print(species_list)
    
    # Step 3: Build final dataset
    print("\n🏗️ Step 3: Building final dataset...")
    builder = DatasetBuilder()
    fish_mapping_df = builder.create_template_dataset()
    
    print(f"\nCreated dataset with {len(fish_mapping_df)} records")
    print("\nSample data:")
    print(fish_mapping_df.head())
    
    # Save dataset
    builder.save_dataset(fish_mapping_df)
    
    print("\n" + "=" * 60)
    print("✅ Pipeline Complete!")
    print(f"\n📁 Output files:")
    print(f"   - datasets/processed/ (extracted text and tables)")
    print(f"   - data/final/fish_mapping.csv (main dataset)")
    print(f"   - data/final/fish_mapping.json (JSON version)")
    print("\n🎯 Next steps:")
    print("   1. Review extracted data in datasets/processed/")
    print("   2. Manually enrich fish_mapping.csv with PDF data")
    print("   3. Train ML model: python training/scripts/train_model.py")


if __name__ == "__main__":
    main()