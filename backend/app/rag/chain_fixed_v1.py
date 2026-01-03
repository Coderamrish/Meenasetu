# chain_fixed_v1.py - For LangChain 1.2.0+
import os
import sys
from pathlib import Path

# Fix path
BASE_DIR = Path(__file__).parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

# CORRECT imports for LangChain 1.x
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# Configuration
VECTOR_DB_PATH = BASE_DIR / "models" / "vector_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def initialize_rag_v1():
    """Initialize RAG for LangChain 1.x"""
    print("🚀 Initializing RAG (LangChain 1.x)...")
    
    # 1. Check vector DB
    if not VECTOR_DB_PATH.exists():
        raise FileNotFoundError(f"Vector DB not found: {VECTOR_DB_PATH}")
    
    # 2. Load embeddings and vector store
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    print(f"📂 Loading vector DB from: {VECTOR_DB_PATH}")
    vectorstore = Chroma(
        persist_directory=str(VECTOR_DB_PATH),
        embedding_function=embeddings
    )
    
    doc_count = vectorstore._collection.count()
    print(f"✅ Loaded {doc_count} documents")
    
    # 3. Initialize LLM (use tiny model for testing)
    print("🤖 Loading LLM...")
    llm_pipeline = pipeline(
        "text-generation",
        model="microsoft/phi-2",  # Small, fast model
        max_new_tokens=200,
        temperature=0.7,
        device=-1  # CPU
    )
    
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    
    # 4. Create prompt
    template = """You are MeenaSetu AI, an expert fisheries assistant.
    Use the context to answer the question.
    
    Context: {context}
    
    Question: {input}
    
    Answer: """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 5. Create chains
    print("🔗 Creating RAG chain...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # For LangChain 1.x, use this syntax:
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    print("✅ RAG system ready!")
    return rag_chain, vectorstore

def ask_question_v1(rag_chain, question: str):
    """Ask a question"""
    try:
        print(f"\n❓ Question: {question}")
        result = rag_chain.invoke({"input": question})
        
        answer = result.get("answer", "No answer")
        context = result.get("context", [])
        
        sources = []
        for doc in context:
            sources.append({
                "content": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                "source": doc.metadata.get("source", "Unknown")
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "total_sources": len(sources)
        }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "error": True
        }

# Test
if __name__ == "__main__":
    try:
        rag_chain, vectorstore = initialize_rag_v1()
        
        # Test questions
        questions = [
            "What are freshwater fish?",
            "Tell me about catfish",
        ]
        
        for question in questions:
            result = ask_question_v1(rag_chain, question)
            print(f"\n🤖 Answer: {result['answer']}")
            if result['sources']:
                print(f"📚 Sources ({result['total_sources']}):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. {source['source']}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Try this alternative instead...")