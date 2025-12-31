"""
Quick Start Script for MeenaSetu
Run this first to set up and test extraction
"""

import os
import sys
from pathlib import Path

def check_environment():
    """Check if environment is set up correctly"""
    print("🔍 Checking environment...")
    
    checks = {
        'Python version': sys.version_info >= (3, 8),
        'datasets folder': Path('datasets').exists(),
        'PDFs found': len(list(Path('datasets').glob('*.pdf'))) > 0,
        'processed folder': True,  # Will create if needed
        'data/final folder': True  # Will create if needed
    }
    
    all_ok = True
    for check, status in checks.items():
        icon = '✅' if status else '❌'
        print(f"  {icon} {check}")
        if not status:
            all_ok = False
    
    return all_ok

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    dirs = [
        'datasets/processed',
        'data/final',
        'data/processed/csv',
        'scripts'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {dir_path}")

def check_dependencies():
    """Check if required packages are installed"""
    print("\n📦 Checking dependencies...")
    
    required = {
        'pandas': 'pip install pandas',
        'PyPDF2': 'pip install PyPDF2',
        'pdfplumber': 'pip install pdfplumber'
    }
    
    missing = []
    for package, install_cmd in required.items():
        try:
            __import__(package.lower())
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - Not installed")
            missing.append(install_cmd)
    
    if missing:
        print("\n⚠️  Missing packages! Install with:")
        for cmd in missing:
            print(f"  {cmd}")
        return False
    
    return True

def count_pdfs():
    """Count and list PDFs"""
    print("\n📄 PDFs in datasets folder:")
    
    pdf_files = list(Path('datasets').glob('*.pdf'))
    
    if not pdf_files:
        print("  ⚠️  No PDF files found!")
        print("  Please move your PDFs to the datasets/ folder")
        return []
    
    for pdf in pdf_files[:10]:  # Show first 10
        size_mb = pdf.stat().st_size / (1024 * 1024)
        print(f"  📄 {pdf.name} ({size_mb:.1f} MB)")
    
    if len(pdf_files) > 10:
        print(f"  ... and {len(pdf_files) - 10} more files")
    
    print(f"\n  Total: {len(pdf_files)} PDF files")
    return pdf_files

def test_extraction_on_one_pdf():
    """Test extraction on one PDF"""
    try:
        import PyPDF2
        import pdfplumber
        
        pdf_files = list(Path('datasets').glob('*.pdf'))
        if not pdf_files:
            return False
        
        test_pdf = pdf_files[0]
        print(f"\n🧪 Testing extraction on: {test_pdf.name}")
        
        # Test PyPDF2
        try:
            with open(test_pdf, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                num_pages = len(reader.pages)
                first_page_text = reader.pages[0].extract_text()[:200]
                
                print(f"  ✅ PyPDF2: {num_pages} pages")
                print(f"  Preview: {first_page_text[:100]}...")
        except Exception as e:
            print(f"  ⚠️  PyPDF2 error: {e}")
        
        # Test pdfplumber
        try:
            with pdfplumber.open(test_pdf) as pdf:
                num_tables = sum(len(page.extract_tables()) for page in pdf.pages)
                print(f"  ✅ pdfplumber: {num_tables} tables found")
        except Exception as e:
            print(f"  ⚠️  pdfplumber error: {e}")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Cannot test extraction: {e}")
        print("  Install missing packages first!")
        return False

def show_next_steps():
    """Show next steps"""
    print("\n" + "="*60)
    print("🎯 Next Steps:")
    print("="*60)
    
    steps = [
        "1. Install missing packages (if any shown above)",
        "2. Run full extraction pipeline:",
        "   python scripts\\extract_pdf_data.py",
        "",
        "3. Review extracted data:",
        "   dir datasets\\processed",
        "",
        "4. Open and enrich dataset:",
        "   code data\\final\\fish_mapping.csv",
        "",
        "5. Train ML model:",
        "   python training\\scripts\\train_model.py"
    ]
    
    for step in steps:
        print(f"  {step}")
    
    print("\n" + "="*60)

def main():
    """Main execution"""
    print("\n" + "="*60)
    print("🐟 MeenaSetu - Quick Start Setup")
    print("="*60 + "\n")
    
    # Check environment
    if not check_environment():
        print("\n⚠️  Environment check failed!")
        print("Make sure you're in the Meenasetu directory")
        return
    
    # Create directories
    create_directories()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Count PDFs
    pdf_files = count_pdfs()
    
    # Test extraction if dependencies OK
    if deps_ok and pdf_files:
        test_extraction_on_one_pdf()
    
    # Show next steps
    show_next_steps()
    
    print("\n✅ Setup complete! Ready to extract data.")

if __name__ == "__main__":
    main()