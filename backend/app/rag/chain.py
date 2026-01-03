import os
import json
import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image
import io
import base64
from collections import defaultdict
from dotenv import load_dotenv

# PyTorch for ML image classification
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Load .env from correct location
possible_env_paths = [
    Path(__file__).parent.parent / ".env",
    Path(__file__).parent.parent.parent / ".env",
    Path(__file__).parent / ".env",
]

env_loaded = False
for env_path in possible_env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded .env from: {env_path}")
        env_loaded = True
        break

if not env_loaded:
    print("⚠️ Warning: .env file not found. Using default configuration.")

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# ============================================================
# 📋 LOGGING SETUP
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('meenasetu.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# 🔧 ENHANCED CONFIGURATION
# ============================================================
class Config:
    SCRIPT_DIR = Path(__file__).resolve().parent
    
    if (SCRIPT_DIR.parent.parent / ".env").exists():
        BASE_DIR = SCRIPT_DIR.parent.parent
    elif (SCRIPT_DIR.parent / ".env").exists():
        BASE_DIR = SCRIPT_DIR.parent
    else:
        BASE_DIR = SCRIPT_DIR.parent
    
    VECTOR_DB_DIR = BASE_DIR / "models" / "vector_db"
    UPLOADS_DIR = BASE_DIR / "uploads"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    CACHE_DIR = BASE_DIR / "cache"
    TRAINING_DIR = BASE_DIR.parent / "training" if (BASE_DIR.parent / "training").exists() else BASE_DIR / "training"
    
    PYTORCH_MODEL_PATH = TRAINING_DIR / "checkpoints" / "best_model.pth"
    CLASS_MAPPING_PATH = TRAINING_DIR / "checkpoints" / "class_mapping.json"
    
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    if not GROQ_API_KEY:
        logger.warning("⚠️ GROQ_API_KEY not set! LLM features will not work.")
    
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    RETRIEVAL_K = 7
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def setup_directories():
        for directory in [Config.VECTOR_DB_DIR, Config.UPLOADS_DIR, 
                         Config.OUTPUTS_DIR, Config.CACHE_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def validate_setup():
        issues = []
        
        if not Config.GROQ_API_KEY:
            issues.append("❌ GROQ_API_KEY not set")
        
        if not Config.VECTOR_DB_DIR.exists():
            issues.append(f"⚠️ Vector DB directory not found: {Config.VECTOR_DB_DIR}")
        
        if issues:
            print("\n⚠️ Configuration Issues Found:")
            for issue in issues:
                print(f"   {issue}")
            return False
        
        print("✅ Configuration validated successfully!")
        return True

Config.setup_directories()

# ============================================================
# 🤖 FISH CLASSIFICATION MODEL
# ============================================================
class FishClassificationModel:
    def __init__(self, model_path: Optional[str] = None, class_mapping_path: Optional[str] = None):
        self.device = Config.DEVICE
        self.model = None
        self.class_mapping = {}
        self.transforms = None
        self.is_loaded = False
        
        logger.info(f"🤖 Initializing Fish Classification Model on {self.device}")
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path, class_mapping_path)
        else:
            logger.warning("⚠️ ML model not found.")
    
    def load_model(self, model_path: str, class_mapping_path: Optional[str] = None):
        try:
            logger.info(f"📥 Loading model from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
                
                if 'species_mapping' in checkpoint:
                    species_mapping = checkpoint['species_mapping']
                    if isinstance(species_mapping, dict):
                        if 'species_to_id' in species_mapping:
                            self.class_mapping = species_mapping['species_to_id']
                        elif 'id_to_species' in species_mapping:
                            self.class_mapping = {species: int(idx) for idx, species in species_mapping['id_to_species'].items()}
                        else:
                            self.class_mapping = species_mapping
            
            if class_mapping_path and Path(class_mapping_path).exists():
                with open(class_mapping_path, 'r', encoding='utf-8') as f:
                    external_mapping = json.load(f)
                    if 'species_to_id' in external_mapping:
                        self.class_mapping = external_mapping['species_to_id']
                    elif 'id_to_species' in external_mapping:
                        self.class_mapping = {species: int(idx) for idx, species in external_mapping['id_to_species'].items()}
            
            num_classes = self._detect_num_classes(state_dict)
            self.model = self._create_model_architecture(num_classes)
            
            try:
                self.model.load_state_dict(state_dict, strict=True)
            except:
                remapped = {f'backbone.{k}' if not k.startswith('backbone.') else k: v 
                           for k, v in state_dict.items()}
                self.model.load_state_dict(remapped, strict=False)
            
            self.model.to(self.device)
            self.model.eval()
            self._setup_transforms()
            
            self.is_loaded = True
            logger.info(f"✅ Model loaded: {num_classes} classes, {len(self.class_mapping)} species mapped")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            self.is_loaded = False
    
    def _detect_num_classes(self, state_dict: Dict) -> int:
        for key in ['backbone.classifier.9.weight', 'classifier.9.weight', 
                    'backbone.classifier.1.weight', 'classifier.1.weight']:
            if key in state_dict:
                return state_dict[key].shape[0]
        return len(self.class_mapping) if self.class_mapping else 14
    
    def _create_model_architecture(self, num_classes: int) -> nn.Module:
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        num_features = model.classifier[1].in_features
        
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(256, num_classes)
        )
        
        class FishClassifier(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
            def forward(self, x):
                return self.backbone(x)
        
        return FishClassifier(model)
    
    def _setup_transforms(self):
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def classify_image(self, image_path: str) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"status": "no_model", "message": "ML model not loaded"}
        
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transforms(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = predicted.item()
            confidence_score = confidence.item()
            
            idx_to_class = {v: k for k, v in self.class_mapping.items()}
            species_name = idx_to_class.get(predicted_class, f"Unknown_Species_{predicted_class}")
            
            top3_prob, top3_indices = torch.topk(probabilities, min(3, probabilities.size(1)))
            top3_predictions = []
            for prob, idx in zip(top3_prob[0], top3_indices[0]):
                name = idx_to_class.get(idx.item(), f"Unknown_{idx.item()}")
                top3_predictions.append({"species": name, "confidence": prob.item()})
            
            return {
                "status": "success",
                "predicted_species": species_name,
                "confidence": confidence_score,
                "top3_predictions": top3_predictions
            }
        except Exception as e:
            logger.error(f"❌ Classification error: {e}")
            return {"status": "error", "message": str(e)}

# ============================================================
# 🎯 ENHANCED DOCUMENT PROCESSOR
# ============================================================
class AdvancedDocumentProcessor:
    def __init__(self, fish_classifier: Optional[FishClassificationModel] = None):
        self.file_stats = defaultdict(int)
        self.processing_history = []
        self.fish_classifier = fish_classifier
        logger.info("📄 Document Processor initialized")
    
    def load_pdf(self, file_path: str) -> List[Document]:
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    "type": "pdf",
                    "source": str(file_path),
                    "page": i + 1,
                    "filename": Path(file_path).name,
                    "loaded_at": datetime.now().isoformat()
                })
            
            logger.info(f"✅ PDF loaded: {Path(file_path).name} ({len(documents)} pages)")
            self.file_stats["pdf"] += len(documents)
            return documents
        except Exception as e:
            logger.error(f"❌ PDF error: {e}")
            return []
    
    def load_csv(self, file_path: str) -> List[Document]:
        try:
            df = pd.read_csv(file_path)
            
            if df.empty:
                logger.warning(f"⚠️ Empty CSV: {file_path}")
                return []
            
            # Simple content without revealing structure
            content_parts = [
                f"Data from {Path(file_path).name}",
                f"\nThe dataset contains information about: {', '.join(df.columns.tolist())}",
                f"\nKey statistics and insights:",
            ]
            
            # Add meaningful insights
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    content_parts.append(f"- {col}: ranges from {df[col].min()} to {df[col].max()}, average {df[col].mean():.2f}")
                else:
                    unique_count = df[col].nunique()
                    if unique_count <= 10:
                        content_parts.append(f"- {col}: includes {', '.join(map(str, df[col].unique().tolist()))}")
                    else:
                        content_parts.append(f"- {col}: contains {unique_count} different values")
            
            # Add sample data without structure
            content_parts.append(f"\nSample entries:")
            for _, row in df.head(5).iterrows():
                row_str = ", ".join([f"{col}: {val}" for col, val in row.items()])
                content_parts.append(f"  • {row_str}")
            
            doc = Document(
                page_content="\n".join(content_parts),
                metadata={
                    "type": "csv",
                    "source": str(file_path),
                    "filename": Path(file_path).name,
                    "loaded_at": datetime.now().isoformat()
                }
            )
            
            logger.info(f"✅ CSV loaded: {Path(file_path).name}")
            self.file_stats["csv"] += 1
            return [doc]
        except Exception as e:
            logger.error(f"❌ CSV error: {e}")
            return []
    
    def load_json(self, file_path: str) -> List[Document]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                content = f"Data from {Path(file_path).name}\n\nContains {len(data)} records with information about various attributes."
                if data:
                    content += f"\n\nSample data:\n{json.dumps(data[:3], indent=2, ensure_ascii=False)}"
            else:
                content = f"Data from {Path(file_path).name}\n\n{json.dumps(data, indent=2, ensure_ascii=False)}"
            
            doc = Document(
                page_content=content,
                metadata={
                    "type": "json",
                    "source": str(file_path),
                    "filename": Path(file_path).name,
                    "loaded_at": datetime.now().isoformat()
                }
            )
            
            logger.info(f"✅ JSON loaded: {Path(file_path).name}")
            self.file_stats["json"] += 1
            return [doc]
        except Exception as e:
            logger.error(f"❌ JSON error: {e}")
            return []
    
    def load_txt(self, file_path: str) -> List[Document]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            doc = Document(
                page_content=content,
                metadata={
                    "type": "txt",
                    "source": str(file_path),
                    "filename": Path(file_path).name,
                    "loaded_at": datetime.now().isoformat()
                }
            )
            
            logger.info(f"✅ TXT loaded: {Path(file_path).name}")
            self.file_stats["txt"] += 1
            return [doc]
        except Exception as e:
            logger.error(f"❌ TXT error: {e}")
            return []
    
    def load_image(self, file_path: str) -> List[Document]:
        try:
            img = Image.open(file_path)
            
            content_parts = [
                f"Image analysis: {Path(file_path).name}",
                f"Image dimensions: {img.size[0]} × {img.size[1]} pixels"
            ]
            
            metadata = {
                "type": "image",
                "source": str(file_path),
                "filename": Path(file_path).name,
                "width": img.size[0],
                "height": img.size[1],
                "loaded_at": datetime.now().isoformat(),
                "ml_classified": False
            }
            
            if self.fish_classifier and self.fish_classifier.is_loaded:
                result = self.fish_classifier.classify_image(file_path)
                if result['status'] == 'success':
                    content_parts.extend([
                        f"\nIdentified species: {result['predicted_species']}",
                        f"Confidence level: {result['confidence']:.1%}",
                        "\nAlternative possibilities:"
                    ])
                    for i, pred in enumerate(result['top3_predictions'][1:], 2):
                        content_parts.append(f"  {i}. {pred['species']} ({pred['confidence']:.1%})")
                    
                    metadata.update({
                        "ml_predicted_species": result['predicted_species'],
                        "ml_confidence": result['confidence'],
                        "ml_top3": json.dumps(result['top3_predictions']),
                        "ml_classified": True
                    })
            
            doc = Document(
                page_content="\n".join(content_parts),
                metadata=metadata
            )
            
            logger.info(f"✅ Image loaded: {Path(file_path).name}")
            self.file_stats["image"] += 1
            return [doc]
        except Exception as e:
            logger.error(f"❌ Image error: {e}")
            return []
    
    def process_file(self, file_path: str) -> Tuple[List[Document], Dict]:
        if not Path(file_path).exists():
            return [], {"error": "File not found"}
        
        ext = Path(file_path).suffix.lower()
        processors = {
            '.pdf': self.load_pdf,
            '.csv': self.load_csv,
            '.json': self.load_json,
            '.txt': self.load_txt,
            '.jpg': self.load_image,
            '.jpeg': self.load_image,
            '.png': self.load_image,
            '.gif': self.load_image,
            '.bmp': self.load_image,
        }
        
        if ext not in processors:
            return [], {"error": f"Unsupported file type: {ext}"}
        
        docs = processors[ext](file_path)
        
        return docs, {
            "file": Path(file_path).name,
            "type": ext,
            "documents_created": len(docs),
            "timestamp": datetime.now().isoformat()
        }

# ============================================================
# 🧠 EMBEDDING MANAGER
# ============================================================
class EmbeddingManager:
    def __init__(self):
        logger.info("🧠 Initializing Embedding Manager...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info(f"✅ Embeddings ready: {Config.EMBED_MODEL}")
    
    def get_embeddings(self):
        return self.embeddings

# ============================================================
# 📚 VECTOR DATABASE MANAGER
# ============================================================
class VectorDatabaseManager:
    def __init__(self, embedding_manager: EmbeddingManager):
        logger.info("🗄️ Initializing Vector Database...")
        self.embedding_manager = embedding_manager
        self.vector_db = None
        self.retriever = None
        self.document_count = 0
        self._initialize_db()
    
    def _initialize_db(self):
        try:
            self.vector_db = Chroma(
                persist_directory=str(Config.VECTOR_DB_DIR),
                embedding_function=self.embedding_manager.get_embeddings(),
                collection_name="meenasetu_production",
            )
            self.document_count = self.vector_db._collection.count()
            logger.info(f"✅ Vector DB loaded: {self.document_count} documents")
        except Exception as e:
            logger.warning(f"⚠️ Creating new vector DB: {e}")
            self.vector_db = Chroma(
                embedding_function=self.embedding_manager.get_embeddings(),
                collection_name="meenasetu_production",
            )
        
        self._setup_retriever()
    
    def _setup_retriever(self):
        self.retriever = self.vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": Config.RETRIEVAL_K},
        )
    
    def add_documents(self, documents: List[Document]) -> Dict:
        if not documents:
            return {"status": "error", "message": "No documents"}
        
        try:
            self.vector_db.add_documents(documents)
            self.document_count = self.vector_db._collection.count()
            logger.info(f"✅ Added {len(documents)} documents")
            return {
                "status": "success",
                "documents_added": len(documents),
                "total_documents": self.document_count
            }
        except Exception as e:
            logger.error(f"❌ Add documents error: {e}")
            return {"status": "error", "message": str(e)}
    
    def search(self, query: str) -> List[Document]:
        try:
            return self.retriever.invoke(query)
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []
    
    def get_stats(self) -> Dict:
        return {
            "total_documents": self.document_count,
            "vector_db_dir": str(Config.VECTOR_DB_DIR),
            "collection_name": "meenasetu_production"
        }

# ============================================================
# 💬 GROQ LLM MANAGER WITH MEMORY
# ============================================================
class GroqLLMManager:
    def __init__(self):
        logger.info("🤖 Initializing Groq LLM with Memory...")
        
        if not Config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found!")
        
        try:
            self.llm = ChatGroq(
                api_key=Config.GROQ_API_KEY,
                model=Config.GROQ_MODEL,
                temperature=0.3,  # Lower temperature for reduced hallucination
                max_tokens=2048,
            )
            
            # Conversation memory
            self.conversation_history = []
            self.chat_history = []  # For LangChain memory
            
            logger.info(f"✅ Groq LLM ready: {Config.GROQ_MODEL}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Groq LLM: {e}")
            raise
    
    def get_llm(self):
        return self.llm
    
    def add_to_history(self, role: str, content: str):
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        if role == "user":
            self.chat_history.append(HumanMessage(content=content))
        elif role == "assistant":
            self.chat_history.append(AIMessage(content=content))
    
    def get_history(self) -> List[Dict]:
        return self.conversation_history
    
    def get_chat_messages(self) -> List:
        return self.chat_history[-10:]  # Last 10 messages for context
    
    def clear_history(self):
        self.conversation_history = []
        self.chat_history = []

# ============================================================
# 🎨 VISUALIZATION ENGINE
# ============================================================
class VisualizationEngine:
    def __init__(self):
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 7)
        logger.info("🎨 Visualization Engine ready")
    
    def create_bar_chart(self, data: Dict, title: str, xlabel: str = "Categories", 
                        ylabel: str = "Values") -> str:
        try:
            plt.figure(figsize=(12, 7))
            bars = plt.bar(data.keys(), data.values(), color='steelblue', edgecolor='black', alpha=0.7)
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
            
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
            plt.xlabel(xlabel, fontsize=12, fontweight='bold')
            plt.ylabel(ylabel, fontsize=12, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            return self._save_plot("bar_chart")
        except Exception as e:
            logger.error(f"❌ Bar chart error: {e}")
            return None
    
    def create_pie_chart(self, data: Dict, title: str) -> str:
        try:
            plt.figure(figsize=(10, 8))
            colors = sns.color_palette("husl", len(data))
            wedges, texts, autotexts = plt.pie(
                data.values(), 
                labels=data.keys(), 
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            
            return self._save_plot("pie_chart")
        except Exception as e:
            logger.error(f"❌ Pie chart error: {e}")
            return None
    
    def _save_plot(self, plot_type: str) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{plot_type}_{timestamp}.png"
        filepath = Config.OUTPUTS_DIR / filename
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Plot saved: {filepath}")
        return str(filepath)

# ============================================================
# 🔗 ENHANCED RAG CHAIN WITH MEMORY
# ============================================================
class AdvancedRAGChain:
    def __init__(self, vector_db: VectorDatabaseManager, llm_manager: GroqLLMManager):
        logger.info("🔗 Initializing Advanced RAG Chain with Memory...")
        self.vector_db = vector_db
        self.llm_manager = llm_manager
        self.chain = None
        self._build_chain()
    
    def _build_chain(self):
        system_prompt = """You are MeenaSetu, an expert AI assistant specializing in aquatic biodiversity, fisheries, and aquaculture.
CRITICAL RULES TO PREVENT HALLUCINATION:
1. ONLY use information from the provided context.
2. If context doesn't contain relevant information, use your general knowledge about aquatic science.
3. NEVER mention data structure details (rows, columns, dimensions).
4. NEVER say "data not found" or "no information available".
5. ALWAYS respond naturally, conversationally, and helpfully.
6. For visualization requests, acknowledge and confirm you can help.
7. For image classification requests, acknowledge the request.
8. Use conversation history for follow-up questions.
9. NEVER apologize for limitations.

PERSONALIZATION & GREETING:
- Start each interaction with a time-appropriate greeting based on current local time:
  • Morning (5 AM–12 PM): "Good morning! How can I assist you with aquatic biodiversity or fisheries today?"
  • Afternoon (12 PM–5 PM): "Good afternoon! What aquatic insights can I provide for you today?"
  • Evening (5 PM–9 PM): "Good evening! Ready to explore some fisheries or aquaculture data?"
  • Night (9 PM–5 AM): "Hello! Working late on aquatic science, I see—how can I assist?"
- Maintain a friendly and personalized tone throughout the conversation.

RESPONSE STRATEGY:
- Answer naturally without mentioning sources or technical details.
- Keep responses focused, helpful, and clear.
- If asked about data you don't have, provide accurate general knowledge instead.
- When images are provided, acknowledge the request and provide species classification if possible.
- When numerical or categorical data is present, suggest appropriate visualizations and explain them clearly.
- Maintain context across the conversation to handle follow-ups smoothly.
- Keep a friendly, professional, and conversational tone.

CONTEXT FROM DOCUMENTS:
{context}

CONVERSATION HISTORY:
{chat_history}

CURRENT QUESTION:
{question}

RESPONSE:"""
        
        prompt = ChatPromptTemplate.from_template(system_prompt)
        
        def retrieve_docs(query: str) -> str:
            docs = self.vector_db.search(query)
            if not docs:
                return "Use your general knowledge about aquatic science and fisheries."
            
            context_parts = []
            for doc in docs:
                content = doc.page_content[:800]
                context_parts.append(content)
            
            return "\n\n".join(context_parts)
        
        def format_chat_history(messages) -> str:
            if not messages:
                return "No previous conversation."
            
            formatted = []
            for msg in messages[-6:]:  # Last 6 messages
                if isinstance(msg, HumanMessage):
                    formatted.append(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    formatted.append(f"Assistant: {msg.content}")
            
            return "\n".join(formatted)
        
        self.chain = (
            {
                "context": RunnableLambda(retrieve_docs),
                "chat_history": RunnableLambda(lambda x: format_chat_history(self.llm_manager.get_chat_messages())),
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm_manager.get_llm()
            | StrOutputParser()
        )
    
    def invoke(self, question: str) -> str:
        try:
            logger.info(f"❓ Processing: {question[:80]}...")
            response = self.chain.invoke(question)
            logger.info("✅ Query processed")
            return response
        except Exception as e:
            logger.error(f"❌ Chain error: {e}")
            return "I encountered an error processing your question. Please try rephrasing it."

# ============================================================
# 🌍 MAIN MEENASETU APPLICATION
# ============================================================
class MeenasetuAI:
    """Main MeenaSetu AI Application - Production Ready"""
    
    def __init__(self):
        logger.info("=" * 70)
        logger.info("🐠 MEENASETU AI - INTELLIGENT AQUATIC BIODIVERSITY EXPERT 🐠")
        logger.info("=" * 70)
        
        # Initialize ML model
        model_path = str(Config.PYTORCH_MODEL_PATH) if Config.PYTORCH_MODEL_PATH.exists() else None
        class_mapping_path = str(Config.CLASS_MAPPING_PATH) if Config.CLASS_MAPPING_PATH.exists() else None
        self.fish_classifier = FishClassificationModel(model_path, class_mapping_path)
        
        # Initialize components
        self.doc_processor = AdvancedDocumentProcessor(fish_classifier=self.fish_classifier)
        self.embedding_manager = EmbeddingManager()
        self.vector_db = VectorDatabaseManager(self.embedding_manager)
        self.llm_manager = GroqLLMManager()
        self.rag_chain = AdvancedRAGChain(self.vector_db, self.llm_manager)
        self.viz_engine = VisualizationEngine()
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )
        
        self.session_info = {
            "start_time": datetime.now().isoformat(),
            "documents_processed": 0,
            "queries_processed": 0,
            "images_classified": 0,
        }
        
        logger.info("✅ MeenaSetu AI fully initialized!")
        logger.info(f"🤖 ML Model: {'Loaded' if self.fish_classifier.is_loaded else 'Not Available'}")
        logger.info(f"📚 Vector DB: {self.vector_db.document_count} documents")
    
    def upload_file(self, file_path: str) -> Dict:
        """Upload and process any file type"""
        logger.info(f"📥 Uploading: {file_path}")
        
        docs, metadata = self.doc_processor.process_file(file_path)
        
        if not docs:
            return {"status": "error", "message": metadata.get("error", "Unknown error")}
        
        # Check if it's an image with ML classification
        is_ml_image = any(doc.metadata.get("ml_classified", False) for doc in docs)
        if is_ml_image:
            self.session_info["images_classified"] += 1
        
        # Split documents
        split_docs = self.text_splitter.split_documents(docs)
        
        # Add to vector DB
        result = self.vector_db.add_documents(split_docs)
        
        if result["status"] == "success":
            self.session_info["documents_processed"] += len(split_docs)
        
        return {
            "status": "success",
            "file": Path(file_path).name,
            "file_type": metadata.get("type"),
            "chunks_created": len(split_docs),
            "ml_classified": is_ml_image,
            "message": f"✅ Successfully processed {Path(file_path).name}"
        }
    
    def upload_multiple_files(self, file_paths: List[str]) -> Dict:
        """Upload multiple files"""
        results = []
        for file_path in file_paths:
            result = self.upload_file(file_path)
            results.append(result)
        
        successful = sum(1 for r in results if r["status"] == "success")
        
        return {
            "status": "success",
            "total_files": len(file_paths),
            "successful": successful,
            "failed": len(file_paths) - successful,
            "details": results
        }
    
    def ask(self, question: str, include_sources: bool = False) -> Dict:
        """Ask a question with conversation memory"""
        logger.info(f"❓ Question: {question}")
        
        # Detect if this is a visualization request
        viz_keywords = ['chart', 'graph', 'plot', 'visualize', 'visualization', 'pie', 'bar']
        is_viz_request = any(keyword in question.lower() for keyword in viz_keywords)
        
        # Detect if this is an image classification request
        img_keywords = ['classify', 'identify', 'recognize', 'what fish', 'what species']
        is_img_request = any(keyword in question.lower() for keyword in img_keywords)
        
        # Add to history
        self.llm_manager.add_to_history("user", question)
        
        # Get response from RAG chain
        answer = self.rag_chain.invoke(question)
        
        # Get source documents (but don't expose them unless explicitly needed)
        source_docs = self.vector_db.search(question)
        sources = []
        
        if include_sources and source_docs:
            for doc in source_docs[:3]:  # Top 3 sources only
                sources.append({
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "type": doc.metadata.get("type", "Unknown"),
                    "snippet": doc.page_content[:150] + "...",
                    "ml_classified": doc.metadata.get("ml_classified", False),
                    "ml_species": doc.metadata.get("ml_predicted_species", None)
                })
        
        # Add to history
        self.llm_manager.add_to_history("assistant", answer)
        
        self.session_info["queries_processed"] += 1
        
        return {
            "answer": answer,
            "sources": sources,
            "source_count": len(sources),
            "timestamp": datetime.now().isoformat(),
            "has_ml_classification": any(s.get("ml_classified") for s in sources),
            "is_visualization_request": is_viz_request,
            "is_image_request": is_img_request
        }
    
    def classify_image(self, image_path: str) -> Dict:
        """Classify a fish image directly"""
        if not self.fish_classifier.is_loaded:
            return {
                "status": "no_model",
                "message": "ML model not available. Please ensure model is trained and available."
            }
        
        result = self.fish_classifier.classify_image(image_path)
        
        if result.get("status") == "success":
            self.session_info["images_classified"] += 1
        
        return result
    
    def generate_visualization(self, plot_type: str, data: Dict, title: str, 
                              xlabel: str = "", ylabel: str = "") -> Dict:
        """Generate visualization"""
        logger.info(f"📊 Generating {plot_type} chart...")
        
        try:
            if plot_type.lower() in ["bar", "bar_chart"]:
                file_path = self.viz_engine.create_bar_chart(data, title, xlabel, ylabel)
            elif plot_type.lower() in ["pie", "pie_chart"]:
                file_path = self.viz_engine.create_pie_chart(data, title)
            else:
                return {"status": "error", "message": f"Unknown plot type: {plot_type}. Use 'bar' or 'pie'."}
            
            if file_path:
                return {
                    "status": "success",
                    "plot_type": plot_type,
                    "file_path": file_path,
                    "filename": Path(file_path).name,
                    "message": f"✅ {plot_type.capitalize()} chart generated successfully"
                }
            else:
                return {"status": "error", "message": "Failed to generate visualization"}
        except Exception as e:
            logger.error(f"❌ Visualization error: {e}")
            return {"status": "error", "message": str(e)}
    
    def analyze_csv_for_visualization(self, csv_path: str) -> Dict:
        """Analyze CSV and suggest visualizations"""
        try:
            df = pd.read_csv(csv_path)
            
            suggestions = []
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Suggest bar charts for categorical vs numeric
            if categorical_cols and numeric_cols:
                for cat_col in categorical_cols[:2]:  # Top 2 categorical
                    for num_col in numeric_cols[:2]:  # Top 2 numeric
                        suggestions.append({
                            "type": "bar",
                            "description": f"Bar chart of {num_col} by {cat_col}",
                            "data": df.groupby(cat_col)[num_col].mean().to_dict(),
                            "title": f"{num_col} by {cat_col}",
                            "xlabel": cat_col,
                            "ylabel": num_col
                        })
            
            # Suggest pie chart for categorical distribution
            if categorical_cols:
                for col in categorical_cols[:2]:
                    value_counts = df[col].value_counts().head(10).to_dict()
                    suggestions.append({
                        "type": "pie",
                        "description": f"Distribution of {col}",
                        "data": value_counts,
                        "title": f"Distribution of {col}"
                    })
            
            return {
                "status": "success",
                "filename": Path(csv_path).name,
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "suggestions": suggestions,
                "message": f"Found {len(suggestions)} visualization suggestions"
            }
        except Exception as e:
            logger.error(f"❌ CSV analysis error: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            "session_info": self.session_info,
            "database_stats": self.vector_db.get_stats(),
            "file_processing_stats": dict(self.doc_processor.file_stats),
            "conversation_history_length": len(self.llm_manager.get_history()),
            "ml_model_status": "loaded" if self.fish_classifier.is_loaded else "not_available",
            "ml_species_count": len(self.fish_classifier.class_mapping) if self.fish_classifier.is_loaded else 0,
        }
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.llm_manager.get_history()
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.llm_manager.clear_history()
        logger.info("🧹 Conversation history cleared")
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Direct search in vector database"""
        docs = self.vector_db.search(query)
        
        results = []
        for doc in docs[:k]:
            results.append({
                "content": doc.page_content[:500],
                "metadata": doc.metadata,
                "source": doc.metadata.get("filename", "Unknown")
            })
        
        return results

# ============================================================
# 🧪 DEMONSTRATION & TESTING
# ============================================================
def main():
    """Production-ready demonstration"""
    print("\n" + "=" * 80)
    print("🐠 MEENASETU AI - PRODUCTION DEMONSTRATION 🐠")
    print("=" * 80 + "\n")
    
    # Validate configuration
    print("🔍 Validating Configuration...")
    print("-" * 80)
    if not Config.validate_setup():
        print("\n❌ Configuration validation failed!")
        return
    
    print("\n✅ Configuration valid! Initializing AI...\n")
    
    # Initialize
    try:
        ai = MeenasetuAI()
    except Exception as e:
        print(f"\n❌ Initialization failed: {e}")
        return
    
    # Test conversation with memory
    print("\n💬 TEST: CONVERSATION WITH MEMORY")
    print("-" * 80)
    
    test_queries = [
        "What fish species information do you have?",
        "Which one has the highest protein?",  # Follow-up
        "Compare the top two",  # Follow-up
    ]
    
    for query in test_queries:
        print(f"\n❓ User: {query}")
        result = ai.ask(query)
        print(f"🤖 MeenaSetu: {result['answer'][:200]}...")
    
    # Test visualization
    print("\n\n📊 TEST: VISUALIZATION GENERATION")
    print("-" * 80)
    
    sample_data = {
        "Catfish": 500,
        "Rohu": 800,
        "Carp": 600,
        "Tilapia": 700
    }
    
    viz_result = ai.generate_visualization(
        "bar",
        sample_data,
        "Fish Production Analysis",
        "Species",
        "Production (Tons)"
    )
    
    if viz_result['status'] == 'success':
        print(f"✅ {viz_result['message']}")
        print(f"📁 Saved: {viz_result['filename']}")
    
    # Test statistics
    print("\n\n📈 TEST: SYSTEM STATISTICS")
    print("-" * 80)
    stats = ai.get_statistics()
    print(f"✅ Queries Processed: {stats['session_info']['queries_processed']}")
    print(f"✅ Documents in DB: {stats['database_stats']['total_documents']}")
    print(f"✅ ML Model: {stats['ml_model_status']}")
    
    print("\n" + "=" * 80)
    print("✨ PRODUCTION DEMONSTRATION COMPLETE ✨")
    print("=" * 80 + "\n")
    
    print("🎯 PRODUCTION FEATURES:")
    print("   ✅ Reduced hallucination (lower temperature, strict context)")
    print("   ✅ Conversation memory (follow-up questions)")
    print("   ✅ No technical details in responses")
    print("   ✅ Automatic visualization generation")
    print("   ✅ ML image classification")
    print("   ✅ Graceful error handling")
    print("\n🚀 MeenaSetu AI is production-ready!\n")

if __name__ == "__main__":
    main()