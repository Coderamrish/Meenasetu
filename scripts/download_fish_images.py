"""
MeenaSetu - Automatic Fish Image Downloader
Downloads fish images from iNaturalist and other sources
"""

import requests
import pandas as pd
from pathlib import Path
import time
from tqdm import tqdm
import json

class FishImageDownloader:
    """Download fish images from various sources"""
    
    def __init__(self, output_dir='datasets/training/images'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {'downloaded': 0, 'skipped': 0, 'errors': 0}
    
    def download_from_inaturalist(self, species_name, max_images=30, quality='research'):
        """
        Download images from iNaturalist
        
        Args:
            species_name: Scientific name (e.g., "Labeo rohita")
            max_images: Maximum images to download
            quality: 'research' (verified) or 'needs_id' (unverified)
        """
        print(f"\n🔍 Searching iNaturalist for: {species_name}")
        
        # Create species folder
        folder_name = species_name.replace(' ', '_')
        species_dir = self.output_dir / folder_name
        species_dir.mkdir(exist_ok=True)
        
        # Skip if already has enough images
        existing_images = list(species_dir.glob('*.jpg'))
        if len(existing_images) >= max_images:
            print(f"   ⏭️  Already has {len(existing_images)} images, skipping")
            self.stats['skipped'] += 1
            return
        
        # Search iNaturalist
        url = "https://api.inaturalist.org/v1/observations"
        params = {
            'taxon_name': species_name,
            'quality_grade': quality,
            'photos': 'true',
            'per_page': max_images,
            'order': 'desc',
            'order_by': 'votes',
            'license': 'cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa,cc0'  # Only CC licenses
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"   ❌ Error fetching data: {e}")
            self.stats['errors'] += 1
            return
        
        if 'results' not in data or not data['results']:
            print(f"   ⚠️  No images found")
            return
        
        # Download images
        downloaded = 0
        for i, obs in enumerate(tqdm(data['results'], desc="   Downloading")):
            if downloaded >= max_images:
                break
            
            if 'photos' not in obs or not obs['photos']:
                continue
            
            # Get medium-sized image
            photo = obs['photos'][0]
            photo_url = photo['url'].replace('square', 'medium')
            
            # Generate filename
            obs_id = obs['id']
            filename = f"{folder_name}_{obs_id}.jpg"
            filepath = species_dir / filename
            
            # Skip if exists
            if filepath.exists():
                continue
            
            # Download
            try:
                img_response = requests.get(photo_url, timeout=10)
                img_response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    f.write(img_response.content)
                
                downloaded += 1
                time.sleep(0.5)  # Be nice to API
                
            except Exception as e:
                print(f"   ⚠️  Error downloading image {i+1}: {e}")
                continue
        
        print(f"   ✅ Downloaded {downloaded} images to {species_dir}")
        self.stats['downloaded'] += downloaded
    
    def download_for_top_species(self, csv_path, num_species=20, images_per_species=30):
        """Download images for top N species from database"""
        print("\n" + "="*70)
        print("🐟 MeenaSetu - Automatic Fish Image Downloader")
        print("="*70)
        print(f"\nSource: iNaturalist.org")
        print(f"Target: {num_species} species, {images_per_species} images each")
        print(f"Output: {self.output_dir}")
        
        # Load fish database
        fish_df = pd.read_csv(csv_path)
        
        # Filter species with complete data
        complete_species = fish_df[
            (fish_df['family'].notna()) & 
            (fish_df['family'] != '') &
            (fish_df['order'].notna()) & 
            (fish_df['order'] != '')
        ]
        
        # Get top species (most common families first)
        top_species = complete_species.head(num_species)['scientific_name'].tolist()
        
        print(f"\n📋 Species to download:")
        for i, species in enumerate(top_species, 1):
            print(f"   {i:2d}. {species}")
        
        input("\n⏸️  Press Enter to start downloading...")
        
        # Download each species
        for i, species in enumerate(top_species, 1):
            print(f"\n[{i}/{len(top_species)}]")
            self.download_from_inaturalist(species, max_images=images_per_species)
            time.sleep(1)  # Rate limiting
        
        # Print summary
        print("\n" + "="*70)
        print("📊 Download Summary")
        print("="*70)
        print(f"   Downloaded: {self.stats['downloaded']} images")
        print(f"   Skipped: {self.stats['skipped']} species (already complete)")
        print(f"   Errors: {self.stats['errors']}")
        
        # Verify dataset
        self.verify_dataset()
    
    def verify_dataset(self):
        """Verify downloaded dataset"""
        print("\n" + "="*70)
        print("🔍 Dataset Verification")
        print("="*70)
        
        species_folders = [d for d in self.output_dir.iterdir() if d.is_dir()]
        
        if not species_folders:
            print("   ❌ No species folders found!")
            return
        
        print(f"\n   Total species: {len(species_folders)}")
        print(f"\n   Images per species:")
        
        valid_species = 0
        total_images = 0
        
        for folder in sorted(species_folders):
            images = list(folder.glob('*.jpg')) + list(folder.glob('*.jpeg')) + list(folder.glob('*.png'))
            count = len(images)
            total_images += count
            
            status = "✅" if count >= 10 else "⚠️ "
            if count >= 10:
                valid_species += 1
            
            print(f"   {status} {folder.name:30s}: {count:3d} images")
        
        print(f"\n   Total images: {total_images}")
        print(f"   Species with ≥10 images: {valid_species}/{len(species_folders)}")
        
        if valid_species >= 5:
            print(f"\n   ✅ Ready to train! ({valid_species} species)")
        else:
            print(f"\n   ⚠️  Need more images (minimum 5 species with 10+ images)")
    
    def download_from_fishbase(self, species_name):
        """
        Download reference image from FishBase
        Note: FishBase has limited API, this is for reference only
        """
        # FishBase doesn't have a public image API
        # This would require web scraping with proper permissions
        print(f"⚠️  FishBase image download not implemented")
        print(f"   Visit: https://www.fishbase.org/search.php?q={species_name}")
    
    def create_download_report(self):
        """Create detailed download report"""
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'output_directory': str(self.output_dir),
            'statistics': self.stats,
            'species': []
        }
        
        for species_folder in self.output_dir.iterdir():
            if species_folder.is_dir():
                images = list(species_folder.glob('*.jpg'))
                report['species'].append({
                    'name': species_folder.name,
                    'image_count': len(images),
                    'ready_for_training': len(images) >= 10
                })
        
        # Save report
        report_path = self.output_dir.parent / 'download_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Report saved: {report_path}")


def main():
    """Main download function"""
    
    print("\n🐟 MeenaSetu Image Downloader")
    print("\nOptions:")
    print("1. Download top 10 species (recommended for testing)")
    print("2. Download top 20 species (good starter dataset)")
    print("3. Download top 50 species (comprehensive dataset)")
    print("4. Download specific species")
    print("5. Verify existing dataset")
    
    choice = input("\nChoose option (1-5): ").strip()
    
    downloader = FishImageDownloader()
    
    if choice == '1':
        downloader.download_for_top_species(
            'data/final/merged/fish_mapping_merged_production.csv',
            num_species=10,
            images_per_species=30
        )
    
    elif choice == '2':
        downloader.download_for_top_species(
            'data/final/merged/fish_mapping_merged_production.csv',
            num_species=20,
            images_per_species=30
        )
    
    elif choice == '3':
        downloader.download_for_top_species(
            'data/final/merged/fish_mapping_merged_production.csv',
            num_species=50,
            images_per_species=50
        )
    
    elif choice == '4':
        species = input("Enter scientific name (e.g., Labeo rohita): ").strip()
        num_images = int(input("Number of images (default 30): ") or "30")
        downloader.download_from_inaturalist(species, max_images=num_images)
    
    elif choice == '5':
        downloader.verify_dataset()
    
    else:
        print("Invalid choice!")
        return
    
    # Create report
    if choice in ['1', '2', '3']:
        downloader.create_download_report()
    
    print("\n✅ Done!")
    print("\n📝 Next steps:")
    print("   1. Review downloaded images")
    print("   2. Remove any incorrect/poor quality images")
    print("   3. Run: python training/scripts/train_fish_classifier.py")


if __name__ == "__main__":
    main()