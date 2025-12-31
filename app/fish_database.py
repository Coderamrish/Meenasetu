"""
MeenaSetu - Fish Database API
Easy-to-use API for your fish identification application
"""

import pandas as pd
from typing import List, Dict, Optional
import json
from pathlib import Path

class FishDatabase:
    """
    Main Fish Database API for MeenaSetu
    
    Usage:
        db = FishDatabase()
        
        # Search by name
        fish = db.search_by_name("Rohu")
        
        # Get by scientific name
        fish = db.get_species("Labeo rohita")
        
        # Filter by family
        cyprinids = db.filter_by_family("Cyprinidae")
        
        # Get all species
        all_fish = db.get_all_species()
    """
    
    def __init__(self, data_file: str = "data/final/merged/fish_mapping_merged_production.csv"):
        """Initialize database with production data"""
        self.data_file = data_file
        self.df = pd.read_csv(data_file)
        print(f"✅ Loaded {len(self.df)} fish species")
        
        # Create search indices for fast lookup
        self._build_indices()
    
    def _build_indices(self):
        """Build search indices for faster queries"""
        # Scientific name index
        self.scientific_index = {
            row['scientific_name'].lower(): idx 
            for idx, row in self.df.iterrows()
        }
        
        # Local name index
        self.local_index = {}
        for idx, row in self.df.iterrows():
            if pd.notna(row['local_name']) and row['local_name']:
                self.local_index[row['local_name'].lower()] = idx
        
        # Common name index
        self.common_index = {}
        for idx, row in self.df.iterrows():
            if pd.notna(row['common_name']) and row['common_name']:
                self.common_index[row['common_name'].lower()] = idx
        
        # Family index
        self.family_groups = self.df.groupby('family').groups
        
        # Order index
        self.order_groups = self.df.groupby('order').groups
    
    def get_species(self, scientific_name: str) -> Optional[Dict]:
        """
        Get species by exact scientific name
        
        Args:
            scientific_name: Scientific name (e.g., "Labeo rohita")
        
        Returns:
            Dictionary with species data or None
        """
        idx = self.scientific_index.get(scientific_name.lower())
        if idx is not None:
            return self.df.iloc[idx].to_dict()
        return None
    
    def search_by_name(self, query: str) -> List[Dict]:
        """
        Search by any name (scientific, local, or common)
        
        Args:
            query: Search term
        
        Returns:
            List of matching species
        """
        query_lower = query.lower()
        results = []
        
        # Check scientific name
        for sci_name, idx in self.scientific_index.items():
            if query_lower in sci_name:
                results.append(self.df.iloc[idx].to_dict())
        
        # Check local name
        for local_name, idx in self.local_index.items():
            if query_lower in local_name:
                fish = self.df.iloc[idx].to_dict()
                if fish not in results:
                    results.append(fish)
        
        # Check common name
        for common_name, idx in self.common_index.items():
            if query_lower in common_name:
                fish = self.df.iloc[idx].to_dict()
                if fish not in results:
                    results.append(fish)
        
        return results
    
    def filter_by_family(self, family: str) -> List[Dict]:
        """Get all species in a family"""
        if family in self.family_groups:
            indices = self.family_groups[family]
            return self.df.iloc[indices].to_dict('records')
        return []
    
    def filter_by_order(self, order: str) -> List[Dict]:
        """Get all species in an order"""
        if order in self.order_groups:
            indices = self.order_groups[order]
            return self.df.iloc[indices].to_dict('records')
        return []
    
    def filter_by_iucn_status(self, status: str) -> List[Dict]:
        """Get species by IUCN conservation status"""
        filtered = self.df[self.df['iucn_status'] == status.upper()]
        return filtered.to_dict('records')
    
    def filter_by_habitat(self, habitat: str) -> List[Dict]:
        """Get species by habitat type"""
        filtered = self.df[self.df['habitat'].str.contains(habitat, case=False, na=False)]
        return filtered.to_dict('records')
    
    def get_all_species(self, limit: Optional[int] = None) -> List[Dict]:
        """Get all species (optionally limited)"""
        if limit:
            return self.df.head(limit).to_dict('records')
        return self.df.to_dict('records')
    
    def get_random_species(self, n: int = 1) -> List[Dict]:
        """Get random species for quiz/training"""
        return self.df.sample(n=n).to_dict('records')
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        return {
            'total_species': len(self.df),
            'families': len(self.df['family'].unique()),
            'orders': len(self.df['order'].unique()),
            'with_local_names': len(self.df[self.df['local_name'].notna() & (self.df['local_name'] != '')]),
            'with_sizes': len(self.df[self.df['max_size'].notna() & (self.df['max_size'] != '')]),
            'iucn_distribution': self.df['iucn_status'].value_counts().to_dict(),
            'top_families': self.df['family'].value_counts().head(10).to_dict()
        }
    
    def export_for_app(self, output_file: str = "app_data/fish_database.json"):
        """Export database as JSON for web/mobile apps"""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'metadata': {
                'total_species': len(self.df),
                'last_updated': pd.Timestamp.now().isoformat(),
                'version': '1.0'
            },
            'species': self.get_all_species()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Exported to {output_file}")
        return output_file


# Example usage and demo functions
def demo_basic_usage():
    """Demonstrate basic database usage"""
    print("\n" + "="*70)
    print("🐟 MeenaSetu Fish Database - Demo")
    print("="*70)
    
    # Initialize database
    db = FishDatabase()
    
    # 1. Get statistics
    print("\n📊 Database Statistics:")
    stats = db.get_statistics()
    print(f"   Total Species: {stats['total_species']}")
    print(f"   Families: {stats['families']}")
    print(f"   Orders: {stats['orders']}")
    
    # 2. Search by name
    print("\n🔍 Search for 'Rohu':")
    results = db.search_by_name("rohu")
    for fish in results[:3]:
        print(f"   • {fish['scientific_name']} ({fish.get('local_name', 'N/A')})")
    
    # 3. Get specific species
    print("\n🐠 Get Labeo rohita:")
    rohu = db.get_species("Labeo rohita")
    if rohu:
        print(f"   Scientific: {rohu['scientific_name']}")
        print(f"   Local: {rohu.get('local_name', 'N/A')}")
        print(f"   Family: {rohu['family']}")
        print(f"   Order: {rohu['order']}")
        print(f"   Size: {rohu.get('max_size', 'N/A')}")
    
    # 4. Filter by family
    print("\n🏷️  Cyprinidae family (first 5):")
    cyprinids = db.filter_by_family("Cyprinidae")
    for fish in cyprinids[:5]:
        print(f"   • {fish['scientific_name']}")
    
    # 5. Filter by IUCN status
    print("\n🔴 Endangered species:")
    endangered = db.filter_by_iucn_status("EN")
    for fish in endangered[:5]:
        print(f"   • {fish['scientific_name']} - {fish.get('local_name', 'N/A')}")
    
    # 6. Random species (for quiz)
    print("\n🎲 Random species:")
    random_fish = db.get_random_species(3)
    for fish in random_fish:
        print(f"   • {fish['scientific_name']}")
    
    # 7. Export for app
    print("\n💾 Exporting for application...")
    db.export_for_app()
    
    print("\n✅ Demo complete!")


def create_flask_app():
    """Create a simple Flask REST API"""
    api_code = '''
from flask import Flask, jsonify, request
from fish_database import FishDatabase

app = Flask(__name__)
db = FishDatabase()

@app.route('/api/species', methods=['GET'])
def get_all_species():
    """Get all species with optional limit"""
    limit = request.args.get('limit', type=int)
    species = db.get_all_species(limit=limit)
    return jsonify({'count': len(species), 'species': species})

@app.route('/api/species/<scientific_name>', methods=['GET'])
def get_species(scientific_name):
    """Get specific species"""
    species = db.get_species(scientific_name)
    if species:
        return jsonify(species)
    return jsonify({'error': 'Species not found'}), 404

@app.route('/api/search', methods=['GET'])
def search():
    """Search species by name"""
    query = request.args.get('q', '')
    results = db.search_by_name(query)
    return jsonify({'count': len(results), 'results': results})

@app.route('/api/family/<family_name>', methods=['GET'])
def get_family(family_name):
    """Get all species in a family"""
    species = db.filter_by_family(family_name)
    return jsonify({'count': len(species), 'species': species})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get database statistics"""
    return jsonify(db.get_statistics())

@app.route('/api/random', methods=['GET'])
def get_random():
    """Get random species"""
    n = request.args.get('n', default=1, type=int)
    species = db.get_random_species(n)
    return jsonify({'species': species})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
'''
    
    with open('app/api.py', 'w') as f:
        f.write(api_code)
    
    print("✅ Created Flask API: app/api.py")
    print("\n🚀 Run with: python app/api.py")
    print("\n📡 Endpoints:")
    print("   GET  /api/species - Get all species")
    print("   GET  /api/species/<name> - Get specific species")
    print("   GET  /api/search?q=rohu - Search by name")
    print("   GET  /api/family/<name> - Get family")
    print("   GET  /api/stats - Get statistics")
    print("   GET  /api/random?n=5 - Get random species")


def create_streamlit_app():
    """Create a simple Streamlit web interface"""
    app_code = '''
import streamlit as st
import pandas as pd
from fish_database import FishDatabase

# Page config
st.set_page_config(
    page_title="MeenaSetu - Fish Database",
    page_icon="🐟",
    layout="wide"
)

# Initialize database
@st.cache_resource
def load_database():
    return FishDatabase()

db = load_database()

# Title
st.title("🐟 MeenaSetu - West Bengal Fish Database")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🔍 Search & Filter")
    
    search_query = st.text_input("Search by name:", "")
    
    st.subheader("Filter by:")
    family_filter = st.selectbox("Family", ["All"] + list(db.df['family'].unique()))
    iucn_filter = st.selectbox("IUCN Status", ["All", "LC", "NT", "VU", "EN", "CR", "DD"])
    
    st.markdown("---")
    st.subheader("📊 Statistics")
    stats = db.get_statistics()
    st.metric("Total Species", stats['total_species'])
    st.metric("Families", stats['families'])
    st.metric("Orders", stats['orders'])

# Main content
if search_query:
    st.header(f"🔍 Search Results: '{search_query}'")
    results = db.search_by_name(search_query)
    if results:
        for fish in results:
            with st.expander(f"{fish['scientific_name']} - {fish.get('local_name', 'N/A')}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Family:** {fish['family']}")
                    st.write(f"**Order:** {fish['order']}")
                    st.write(f"**Local Name:** {fish.get('local_name', 'N/A')}")
                with col2:
                    st.write(f"**Max Size:** {fish.get('max_size', 'N/A')}")
                    st.write(f"**Habitat:** {fish['habitat']}")
                    st.write(f"**IUCN Status:** {fish['iucn_status']}")
    else:
        st.warning("No species found")
else:
    # Filter data
    data = db.df.copy()
    if family_filter != "All":
        data = data[data['family'] == family_filter]
    if iucn_filter != "All":
        data = data[data['iucn_status'] == iucn_filter]
    
    st.header(f"📋 Species List ({len(data)} species)")
    
    # Display table
    display_cols = ['scientific_name', 'local_name', 'family', 'order', 'max_size', 'iucn_status']
    st.dataframe(data[display_cols], use_container_width=True)
    
    # Charts
    st.subheader("📊 Visualizations")
    col1, col2 = st.columns(2)
    
    with col1:
        st.bar_chart(data['family'].value_counts().head(10))
        st.caption("Top 10 Families")
    
    with col2:
        st.bar_chart(data['iucn_status'].value_counts())
        st.caption("IUCN Status Distribution")
'''
    
    Path('app').mkdir(exist_ok=True)
    with open('app/streamlit_app.py', 'w') as f:
        f.write(app_code)
    
    print("✅ Created Streamlit app: app/streamlit_app.py")
    print("\n🚀 Run with: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    # Run demo
    demo_basic_usage()
    
    # Create sample apps
    print("\n" + "="*70)
    print("🚀 Creating Sample Applications")
    print("="*70)
    
    Path('app').mkdir(exist_ok=True)
    
    # Copy main database class to app directory
    import shutil
    shutil.copy(__file__, 'app/fish_database.py')
    print("✅ Copied database API to app/")
    
    # Create Flask API
    create_flask_app()
    
    # Create Streamlit app
    create_streamlit_app()
    
    print("\n" + "="*70)
    print("✅ INTEGRATION COMPLETE!")
    print("="*70)
    print("\n📁 Files created:")
    print("   app/fish_database.py - Core database API")
    print("   app/api.py - Flask REST API")
    print("   app/streamlit_app.py - Web interface")
    print("   app_data/fish_database.json - JSON export")
    
    print("\n🚀 Quick Start:")
    print("   1. Flask API: python app/api.py")
    print("   2. Streamlit: streamlit run app/streamlit_app.py")
    print("   3. Python: from fish_database import FishDatabase")