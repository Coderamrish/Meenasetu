import os
import time
from typing import List, Dict

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_community.llms import Ollama  # or use OpenAI, etc.

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
VECTOR_DB_DIR = os.path.join(BASE_DIR, "models", "vector_db")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# ----------------------------------------

class FastFishQuery:
    def __init__(self, collection_name="langchain"):
        """Initialize with caching for speed"""
        print("🚀 Initializing fast fish query system...")
        start_time = time.time()
        
        # Load embeddings (cache the model)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load vector store
        self.db = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        
        # Create retriever with optimizations
        self.retriever = self.db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 10,  # Increased for better recall
                "score_threshold": 0.3  # Filter out low-quality matches
            }
        )
        
        print(f"✅ Loaded vector DB with {self.db._collection.count()} chunks")
        print(f"⏱️ Initialization time: {time.time() - start_time:.2f}s")
    
    def fast_search(self, query: str, k: int = 5, filters: Dict = None) -> List[Dict]:
        """Ultra-fast semantic search"""
        start_time = time.time()
        
        # Use similarity_search_with_score for better performance
        results = self.db.similarity_search_with_score(
            query=query,
            k=k,
            filter=filters
        )
        
        # Format results
        formatted = []
        for doc, score in results:
            formatted.append({
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
                "relevance": self._score_to_percentage(score)
            })
        
        print(f"🔍 Search time: {time.time() - start_time:.2f}s")
        return formatted
    
    def _score_to_percentage(self, score: float) -> str:
        """Convert similarity score to percentage"""
        # Cosine similarity: 1 = identical, 0 = orthogonal
        # We have distance, so convert
        similarity = 1 - score  # approximate conversion
        percentage = max(0, min(100, similarity * 100))
        return f"{percentage:.1f}%"
    
    def search_by_family(self, family_name: str, query: str = None, k: int = 5):
        """Search within a specific fish family"""
        filters = {"type": "csv"}  # Adjust based on your metadata
        
        if query:
            search_query = f"{query} family:{family_name}"
        else:
            search_query = f"fish family:{family_name}"
        
        return self.fast_search(search_query, k=k, filters=filters)
    
    def find_similar_species(self, species_name: str, k: int = 5):
        """Find species similar to given species"""
        query = f"scientific name: {species_name} characteristics habitat"
        return self.fast_search(query, k=k)
    
    def hybrid_search(self, query: str, k: int = 10):
        """Combine semantic search with keyword filtering"""
        results = self.fast_search(query, k=k*2)  # Get more results
        
        # Simple keyword boosting
        query_keywords = query.lower().split()
        boosted_results = []
        
        for result in results:
            content_lower = result["content"].lower()
            metadata_str = str(result["metadata"]).lower()
            
            # Count keyword matches
            keyword_matches = sum(1 for word in query_keywords 
                                if word in content_lower or word in metadata_str)
            
            # Boost score based on keyword matches
            boosted_score = result["score"] - (keyword_matches * 0.05)
            result["boosted_score"] = boosted_score
            boosted_results.append(result)
        
        # Sort by boosted score
        boosted_results.sort(key=lambda x: x["boosted_score"])
        
        return boosted_results[:k]
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector database"""
        return {
            "total_chunks": self.db._collection.count(),
            "collection_name": self.db._collection.name,
            "embedding_dimension": len(self.embeddings.embed_query("test"))
        }

def interactive_search():
    """Interactive search interface"""
    print("🐟 Fish Knowledge Base - Interactive Search")
    print("=" * 50)
    
    # Initialize once (cached)
    searcher = FastFishQuery()
    
    while True:
        print("\nOptions:")
        print("1. Semantic search")
        print("2. Search by family")
        print("3. Find similar species")
        print("4. Show statistics")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            query = input("Enter search query: ").strip()
            if query:
                results = searcher.fast_search(query, k=5)
                print(f"\n📊 Found {len(results)} results:")
                for i, res in enumerate(results, 1):
                    print(f"\n{i}. [{res['relevance']}]")
                    print(f"   {res['content'][:200]}...")
                    print(f"   Source: {res['metadata'].get('source', 'Unknown')}")
        
        elif choice == "2":
            family = input("Enter fish family (e.g., Cyprinidae): ").strip()
            query = input("Optional specific query: ").strip()
            results = searcher.search_by_family(family, query, k=5)
            print(f"\n🐠 Fish in family {family}:")
            for i, res in enumerate(results, 1):
                print(f"\n{i}. [{res['relevance']}]")
                print(f"   {res['content'][:150]}...")
        
        elif choice == "3":
            species = input("Enter species name: ").strip()
            results = searcher.find_similar_species(species, k=5)
            print(f"\n🔍 Species similar to {species}:")
            for i, res in enumerate(results, 1):
                print(f"\n{i}. [{res['relevance']}]")
                print(f"   {res['content'][:200]}...")
        
        elif choice == "4":
            stats = searcher.get_collection_stats()
            print("\n📈 Vector DB Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        
        elif choice == "5":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")

def batch_test_queries():
    """Test with common fish-related queries"""
    test_queries = [
        "freshwater fish of India",
        "catfish species",
        "endangered fish conservation",
        "fish habitat requirements",
        "commercial fishing species",
        "aquaculture techniques",
        "fish breeding patterns",
        "fish migration behavior",
        "water quality parameters for fish",
        "fish disease symptoms"
    ]
    
    searcher = FastFishQuery()
    
    print("🧪 Running batch query tests...")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = searcher.fast_search(query, k=3)
        
        if results:
            print(f"  Top result: {results[0]['content'][:100]}...")
            print(f"  Relevance: {results[0]['relevance']}")
        else:
            print("  No results found")
    
    print("\n✅ Batch testing complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Query Fish Vector Database")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--family", type=str, help="Filter by fish family")
    parser.add_argument("--species", type=str, help="Find similar species")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--test", action="store_true", help="Run test queries")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_search()
    elif args.test:
        batch_test_queries()
    elif args.species:
        searcher = FastFishQuery()
        results = searcher.find_similar_species(args.species, k=args.k)
        print(f"\n🔍 Species similar to {args.species}:")
        for i, res in enumerate(results, 1):
            print(f"\n{i}. [{res['relevance']}]")
            print(f"   Content: {res['content']}")
            print(f"   Source: {res['metadata'].get('source', 'Unknown')}")
    elif args.family:
        searcher = FastFishQuery()
        results = searcher.search_by_family(args.family, args.query, k=args.k)
        print(f"\n🐠 Fish in family {args.family}:")
        for i, res in enumerate(results, 1):
            print(f"\n{i}. [{res['relevance']}]")
            print(f"   {res['content']}")
    elif args.query:
        searcher = FastFishQuery()
        results = searcher.fast_search(args.query, k=args.k)
        print(f"\n🔍 Results for '{args.query}':")
        for i, res in enumerate(results, 1):
            print(f"\n{i}. [{res['relevance']}]")
            print(f"   Content: {res['content']}")
            print(f"   Source: {res['metadata'].get('source', 'Unknown')}")
            print(f"   Type: {res['metadata'].get('type', 'Unknown')}")
    else:
        # Default: show stats
        searcher = FastFishQuery()
        stats = searcher.get_collection_stats()
        print("📊 Fish Vector Database Status:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print("\nUsage examples:")
        print("  python query_vector_db.py --query \"freshwater fish\"")
        print("  python query_vector_db.py --family \"Cyprinidae\"")
        print("  python query_vector_db.py --interactive")