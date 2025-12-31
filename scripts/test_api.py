"""
MeenaSetu - API Test Client
Test all API endpoints and see responses
"""

import requests
import json
from typing import Dict, List

class FishAPIClient:
    """Client to interact with MeenaSetu Fish API"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        
    def get_all_species(self, limit: int = None) -> Dict:
        """Get all species"""
        params = {'limit': limit} if limit else {}
        response = requests.get(f"{self.base_url}/api/species", params=params)
        return response.json()
    
    def get_species(self, scientific_name: str) -> Dict:
        """Get specific species"""
        response = requests.get(f"{self.base_url}/api/species/{scientific_name}")
        return response.json()
    
    def search(self, query: str) -> Dict:
        """Search by name"""
        response = requests.get(f"{self.base_url}/api/search", params={'q': query})
        return response.json()
    
    def get_family(self, family_name: str) -> Dict:
        """Get species in a family"""
        response = requests.get(f"{self.base_url}/api/family/{family_name}")
        return response.json()
    
    def get_random(self, n: int = 5) -> Dict:
        """Get random species"""
        response = requests.get(f"{self.base_url}/api/random", params={'n': n})
        return response.json()
    
    def get_stats(self) -> Dict:
        """Get statistics"""
        response = requests.get(f"{self.base_url}/api/stats")
        return response.json()


def pretty_print(data: Dict):
    """Pretty print JSON data"""
    print(json.dumps(data, indent=2, ensure_ascii=False))


def demo_all_endpoints():
    """Demonstrate all API endpoints"""
    
    client = FishAPIClient()
    
    print("\n" + "="*70)
    print("🐟 MeenaSetu API - Complete Test Suite")
    print("="*70)
    
    # 1. Get Statistics
    print("\n📊 1. Database Statistics:")
    print("-" * 70)
    stats = client.get_stats()
    print(f"Total Species: {stats['total_species']}")
    print(f"Families: {stats['families']}")
    print(f"Orders: {stats['orders']}")
    print(f"With Local Names: {stats['with_local_names']}")
    print(f"\nTop 5 Families:")
    for family, count in list(stats['top_families'].items())[:5]:
        print(f"  • {family}: {count} species")
    
    # 2. Get All Species (limited)
    print("\n📋 2. Get First 5 Species:")
    print("-" * 70)
    result = client.get_all_species(limit=5)
    for fish in result['species']:
        print(f"  • {fish['scientific_name']} ({fish.get('local_name', 'N/A')})")
    
    # 3. Get Specific Species
    print("\n🐠 3. Get Specific Species (Labeo rohita):")
    print("-" * 70)
    rohu = client.get_species("Labeo rohita")
    if 'error' not in rohu:
        print(f"Scientific Name: {rohu['scientific_name']}")
        print(f"Local Name: {rohu.get('local_name', 'N/A')}")
        print(f"Common Name: {rohu.get('common_name', 'N/A')}")
        print(f"Family: {rohu['family']}")
        print(f"Order: {rohu['order']}")
        print(f"Max Size: {rohu.get('max_size', 'N/A')}")
        print(f"Habitat: {rohu['habitat']}")
        print(f"IUCN Status: {rohu['iucn_status']}")
    
    # 4. Search by Name
    print("\n🔍 4. Search for 'channa' (Snakeheads):")
    print("-" * 70)
    results = client.search("channa")
    print(f"Found {results['count']} species:")
    for fish in results['results'][:5]:
        print(f"  • {fish['scientific_name']} - {fish.get('local_name', 'N/A')}")
    
    # 5. Get Family
    print("\n🏷️  5. Cyprinidae Family (Carps - first 10):")
    print("-" * 70)
    cyprinids = client.get_family("Cyprinidae")
    print(f"Total species in family: {cyprinids['count']}")
    for fish in cyprinids['species'][:10]:
        print(f"  • {fish['scientific_name']}")
    
    # 6. Random Species
    print("\n🎲 6. Random Species (for quiz):")
    print("-" * 70)
    random = client.get_random(3)
    for fish in random['species']:
        print(f"  • {fish['scientific_name']} ({fish.get('local_name', 'N/A')})")
    
    # 7. Search by Local Name
    print("\n🇧🇩 7. Search by Local/Bengali Name 'rui':")
    print("-" * 70)
    results = client.search("rui")
    print(f"Found {results['count']} species:")
    for fish in results['results']:
        print(f"  • {fish['scientific_name']} - {fish.get('local_name', 'N/A')}")
    
    print("\n" + "="*70)
    print("✅ All tests completed successfully!")
    print("="*70)
    
    # Example use cases
    print("\n💡 Example Use Cases:")
    print("-" * 70)
    
    print("\n1️⃣  Fish Identification App:")
    print("   - Use /api/search to find species by name")
    print("   - Display family, size, conservation status")
    
    print("\n2️⃣  Educational Quiz:")
    print("   - Use /api/random to generate questions")
    print("   - Test users on local names, families")
    
    print("\n3️⃣  Conservation Tracker:")
    print("   - Use /api/iucn/EN to get endangered species")
    print("   - Track vulnerable populations")
    
    print("\n4️⃣  Field Guide App:")
    print("   - Browse by family /api/family/Cyprinidae")
    print("   - Filter by habitat, size, status")
    
    print("\n5️⃣  Research Tool:")
    print("   - Use /api/stats for database overview")
    print("   - Export species data for analysis")


def interactive_search():
    """Interactive search demo"""
    client = FishAPIClient()
    
    print("\n" + "="*70)
    print("🔍 Interactive Fish Search")
    print("="*70)
    print("\nType a fish name (scientific, local, or common)")
    print("Examples: rohu, channa, labeo, koi")
    print("Type 'quit' to exit\n")
    
    while True:
        query = input("🐟 Search: ").strip()
        
        if query.lower() == 'quit':
            break
        
        if not query:
            continue
        
        results = client.search(query)
        
        if results['count'] == 0:
            print(f"❌ No species found for '{query}'\n")
        else:
            print(f"\n✅ Found {results['count']} species:\n")
            for i, fish in enumerate(results['results'], 1):
                print(f"{i}. {fish['scientific_name']}")
                print(f"   Local: {fish.get('local_name', 'N/A')}")
                print(f"   Family: {fish['family']}")
                print(f"   Size: {fish.get('max_size', 'N/A')}")
                print(f"   Status: {fish['iucn_status']}")
                print()


def create_fish_quiz():
    """Create a simple fish quiz"""
    client = FishAPIClient()
    
    print("\n" + "="*70)
    print("🎮 Fish Quiz Game")
    print("="*70)
    print("\nGuess the local/Bengali name of these fish!\n")
    
    # Get 5 random fish
    quiz_data = client.get_random(5)
    score = 0
    
    for i, fish in enumerate(quiz_data['species'], 1):
        if not fish.get('local_name'):
            continue
        
        print(f"\nQuestion {i}/5:")
        print(f"Scientific Name: {fish['scientific_name']}")
        print(f"Family: {fish['family']}")
        print(f"Max Size: {fish.get('max_size', 'N/A')}")
        
        answer = input("\n🤔 What is the local name? ").strip().lower()
        correct = fish['local_name'].lower()
        
        if answer == correct:
            print("✅ Correct!")
            score += 1
        else:
            print(f"❌ Wrong! The correct answer is: {fish['local_name']}")
    
    print("\n" + "="*70)
    print(f"🏆 Your Score: {score}/5")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    print("\n🐟 MeenaSetu API Test Client")
    print("\nMake sure the API server is running at http://localhost:5000")
    print("\nChoose an option:")
    print("1. Run full test suite")
    print("2. Interactive search")
    print("3. Play fish quiz")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    try:
        if choice == '1':
            demo_all_endpoints()
        elif choice == '2':
            interactive_search()
        elif choice == '3':
            create_fish_quiz()
        else:
            print("Invalid choice!")
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to API server!")
        print("Make sure the server is running: python app/api.py")
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")