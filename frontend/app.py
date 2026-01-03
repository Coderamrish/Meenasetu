import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Optional, Dict, Any
import base64
from PIL import Image
import io

# ============================================================
# 🎨 PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="🐠 MeenaSetu AI",
    page_icon="🐠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/meenasetu',
        'Report a bug': "https://github.com/yourusername/meenasetu/issues",
        'About': "# MeenaSetu AI\nIntelligent Aquatic Biodiversity Expert System"
    }
)

# ============================================================
# 🎨 CUSTOM CSS STYLING
# ============================================================
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1E88E5;
        --secondary-color: #26C6DA;
        --success-color: #66BB6A;
        --warning-color: #FFA726;
        --error-color: #EF5350;
        --background-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    /* Stats container */
    .stats-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .stat-box {
        flex: 1;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stat-box h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: bold;
    }
    
    .stat-box p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Message styling */
    .user-message {
        background: #E3F2FD;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid #1E88E5;
    }
    
    .ai-message {
        background: #F3E5F5;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid #9C27B0;
    }
    
    /* Source citation */
    .source-citation {
        background: #FFF3E0;
        padding: 0.8rem;
        border-radius: 8px;
        margin-top: 1rem;
        border-left: 3px solid #FF9800;
        font-size: 0.9rem;
    }
    
    /* Upload section */
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    
    /* Success/Error alerts */
    .success-alert {
        background: #C8E6C9;
        color: #2E7D32;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    
    .error-alert {
        background: #FFCDD2;
        color: #C62828;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #F44336;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 🔧 API CONFIGURATION
# ============================================================
API_BASE_URL = "http://localhost:8000"

# ============================================================
# 🛠️ UTILITY FUNCTIONS
# ============================================================

def make_api_request(endpoint: str, method: str = "GET", **kwargs) -> Optional[Dict]:
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, **kwargs)
        elif method == "POST":
            response = requests.post(url, **kwargs)
        else:
            return None
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Error: {str(e)}")
        return None

def format_timestamp(timestamp: str) -> str:
    """Format ISO timestamp to readable format"""
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime("%B %d, %Y at %I:%M %p")
    except:
        return timestamp

def get_file_icon(file_type: str) -> str:
    """Get icon for file type"""
    icons = {
        'pdf': '📄',
        'csv': '📊',
        'json': '📋',
        'txt': '📝',
        'image': '🖼️',
        'jpg': '🖼️',
        'jpeg': '🖼️',
        'png': '🖼️'
    }
    return icons.get(file_type.lower().replace('.', ''), '📁')

def create_metric_card(title: str, value: Any, icon: str):
    """Create a metric card with icon"""
    st.markdown(f"""
    <div class="stat-box fade-in">
        <div style="font-size: 2.5rem;">{icon}</div>
        <h3>{value}</h3>
        <p>{title}</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 📊 VISUALIZATION FUNCTIONS
# ============================================================

def create_interactive_bar_chart(data: Dict, title: str, xlabel: str = "Category", ylabel: str = "Value"):
    """Create interactive bar chart with Plotly"""
    df = pd.DataFrame(list(data.items()), columns=[xlabel, ylabel])
    
    fig = go.Figure(data=[
        go.Bar(
            x=df[xlabel],
            y=df[ylabel],
            marker=dict(
                color=df[ylabel],
                colorscale='Viridis',
                showscale=True
            ),
            text=df[ylabel],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Value: %{y}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=24, color='#2C3E50')),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template='plotly_white',
        height=500,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

def create_interactive_pie_chart(data: Dict, title: str):
    """Create interactive pie chart with Plotly"""
    labels = list(data.keys())
    values = list(data.values())
    
    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            textposition='auto',
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Value: %{value}<br>Percentage: %{percent}<extra></extra>',
            marker=dict(
                colors=px.colors.qualitative.Set3,
                line=dict(color='white', width=2)
            )
        )
    ])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=24, color='#2C3E50')),
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    return fig

def create_line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    """Create interactive line chart"""
    fig = px.line(
        df, 
        x=x_col, 
        y=y_col,
        title=title,
        template='plotly_white',
        markers=True
    )
    
    fig.update_traces(
        line=dict(width=3, color='#667eea'),
        marker=dict(size=10, color='#764ba2')
    )
    
    fig.update_layout(
        title=dict(font=dict(size=24, color='#2C3E50')),
        height=500,
        hovermode='x unified'
    )
    
    return fig

# ============================================================
# 🎯 SESSION STATE INITIALIZATION
# ============================================================

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

if 'stats' not in st.session_state:
    st.session_state.stats = {
        'queries': 0,
        'uploads': 0,
        'visualizations': 0
    }

# ============================================================
# 🏠 MAIN HEADER
# ============================================================

st.markdown("""
<div class="main-header fade-in">
    <h1>🐠 MeenaSetu AI</h1>
    <p>Intelligent Aquatic Biodiversity Expert System</p>
    <p style="font-size: 0.9rem; margin-top: 0.5rem;">
        Powered by RAG, ML Classification & Advanced Analytics
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 📊 DASHBOARD METRICS
# ============================================================

# Fetch system stats
health_data = make_api_request("/health")
stats_data = make_api_request("/stats")

if health_data and stats_data:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card(
            "Documents in DB",
            stats_data.get('statistics', {}).get('database_stats', {}).get('total_documents', 0),
            "📚"
        )
    
    with col2:
        create_metric_card(
            "ML Species",
            stats_data.get('statistics', {}).get('ml_species_count', 0),
            "🤖"
        )
    
    with col3:
        create_metric_card(
            "Queries Processed",
            stats_data.get('statistics', {}).get('session_info', {}).get('queries_processed', 0),
            "❓"
        )
    
    with col4:
        ml_status = stats_data.get('statistics', {}).get('ml_model_status', 'unknown')
        status_icon = "✅" if ml_status == "loaded" else "⚠️"
        create_metric_card(
            "ML Model Status",
            ml_status.upper(),
            status_icon
        )

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# 🎨 MAIN TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Chat & Query",
    "📤 Upload Files",
    "📊 Visualizations",
    "🖼️ Image Classification",
    "📈 Analytics & Stats"
])

# ============================================================
# 💬 TAB 1: CHAT & QUERY
# ============================================================

with tab1:
    st.markdown("### 💬 Ask MeenaSetu AI")
    st.markdown("Ask questions about fish species, aquaculture, or any uploaded documents.")
    
    # Query input
    query = st.text_area(
        "Your Question:",
        placeholder="Example: Which fish species has the highest protein content?",
        height=100,
        key="query_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        ask_button = st.button("🚀 Ask", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🧹 Clear History", use_container_width=True)
    
    if clear_button:
        result = make_api_request("/conversation/clear", method="POST")
        if result:
            st.session_state.conversation_history = []
            st.success("✅ Conversation history cleared!")
            st.rerun()
    
    if ask_button and query:
        with st.spinner("🤔 MeenaSetu is thinking..."):
            result = make_api_request(
                "/rag/query",
                method="POST",
                json={"query": query, "include_sources": True}
            )
            
            if result:
                st.session_state.conversation_history.append({
                    "query": query,
                    "response": result,
                    "timestamp": datetime.now().isoformat()
                })
                st.session_state.stats['queries'] += 1
                st.rerun()
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.markdown("---")
        st.markdown("### 📜 Conversation History")
        
        for i, conv in enumerate(reversed(st.session_state.conversation_history)):
            with st.container():
                # User message
                st.markdown(f"""
                <div class="user-message fade-in">
                    <strong>🙋 You asked:</strong><br>
                    {conv['query']}
                </div>
                """, unsafe_allow_html=True)
                
                # AI response
                response = conv['response']
                st.markdown(f"""
                <div class="ai-message fade-in">
                    <strong>🐠 MeenaSetu:</strong><br>
                    {response['answer']}
                </div>
                """, unsafe_allow_html=True)
                
                # Sources
                if response.get('sources'):
                    with st.expander(f"📚 View {response['source_count']} Sources"):
                        for src in response['sources']:
                            icon = get_file_icon(src['type'])
                            st.markdown(f"""
                            <div class="source-citation">
                                <strong>{icon} {src['filename']}</strong> ({src['type']})<br>
                                <em>{src['snippet'][:200]}...</em>
                                {f"<br>🤖 <strong>ML Classified:</strong> {src['ml_species']}" if src.get('ml_classified') else ''}
                            </div>
                            """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# 📤 TAB 2: UPLOAD FILES
# ============================================================

with tab2:
    st.markdown("### 📤 Upload Documents & Images")
    st.markdown("Upload PDFs, CSVs, JSON, text files, or images for analysis.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'csv', 'json', 'txt', 'jpg', 'jpeg', 'png', 'gif', 'bmp'],
            accept_multiple_files=True,
            key="file_uploader"
        )
    
    with col2:
        st.markdown("**Supported Formats:**")
        st.markdown("""
        - 📄 PDF Documents
        - 📊 CSV Data
        - 📋 JSON Files
        - 📝 Text Files
        - 🖼️ Images (Fish ID)
        """)
    
    if uploaded_files:
        upload_button = st.button("🚀 Upload & Process", type="primary", use_container_width=True)
        
        if upload_button:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, file in enumerate(uploaded_files):
                progress = (idx + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing {file.name}... ({idx + 1}/{len(uploaded_files)})")
                
                # Upload file
                files = {'file': (file.name, file.getvalue(), file.type)}
                result = make_api_request("/upload/file", method="POST", files=files)
                
                if result and result.get('status') == 'success':
                    st.session_state.uploaded_files.append({
                        'filename': file.name,
                        'type': result.get('file_type'),
                        'chunks': result.get('chunks_created'),
                        'ml_classified': result.get('ml_classified', False),
                        'timestamp': datetime.now().isoformat()
                    })
                    st.session_state.stats['uploads'] += 1
                    
                    st.success(f"✅ {file.name} - {result.get('message')}")
                else:
                    st.error(f"❌ Failed to upload {file.name}")
                
                time.sleep(0.2)
            
            progress_bar.progress(1.0)
            status_text.text("✅ All files processed!")
            time.sleep(1)
            st.rerun()
    
    # Display uploaded files
    if st.session_state.uploaded_files:
        st.markdown("---")
        st.markdown("### 📁 Recently Uploaded Files")
        
        df = pd.DataFrame(st.session_state.uploaded_files)
        st.dataframe(
            df[['filename', 'type', 'chunks', 'ml_classified']],
            use_container_width=True,
            hide_index=True
        )

# ============================================================
# 📊 TAB 3: VISUALIZATIONS
# ============================================================

with tab3:
    st.markdown("### 📊 Data Visualizations")
    st.markdown("Create beautiful charts from your data.")
    
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Bar Chart", "Pie Chart", "Line Chart", "Analyze CSV for Suggestions"]
    )
    
    if viz_type in ["Bar Chart", "Pie Chart"]:
        st.markdown("#### Enter Data Manually")
        
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input("Chart Title", "My Chart")
            
        with col2:
            num_entries = st.number_input("Number of data points", 2, 20, 5)
        
        data = {}
        st.markdown("**Enter your data:**")
        
        cols = st.columns(2)
        for i in range(num_entries):
            with cols[i % 2]:
                label = st.text_input(f"Label {i+1}", f"Item {i+1}", key=f"label_{i}")
                value = st.number_input(f"Value {i+1}", 0.0, 10000.0, float(i+1)*10, key=f"value_{i}")
                data[label] = value
        
        if st.button("🎨 Generate Visualization", type="primary"):
            with st.spinner("Creating visualization..."):
                if viz_type == "Bar Chart":
                    fig = create_interactive_bar_chart(data, title)
                else:
                    fig = create_interactive_pie_chart(data, title)
                
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.stats['visualizations'] += 1
                st.success("✅ Visualization created successfully!")
    
    elif viz_type == "Analyze CSV for Suggestions":
        st.markdown("#### Upload CSV for Analysis")
        
        csv_file = st.file_uploader("Choose CSV file", type=['csv'], key="csv_analyzer")
        
        if csv_file:
            if st.button("🔍 Analyze CSV", type="primary"):
                with st.spinner("Analyzing CSV..."):
                    files = {'file': (csv_file.name, csv_file.getvalue(), 'text/csv')}
                    result = make_api_request("/visualize/analyze-csv", method="POST", files=files)
                    
                    if result and result.get('status') == 'success':
                        st.success(f"✅ {result.get('message')}")
                        
                        st.markdown("#### 💡 Suggested Visualizations")
                        
                        for i, suggestion in enumerate(result.get('suggestions', [])):
                            with st.expander(f"Suggestion {i+1}: {suggestion['description']}"):
                                if suggestion['type'] == 'bar':
                                    fig = create_interactive_bar_chart(
                                        suggestion['data'],
                                        suggestion['description']
                                    )
                                else:
                                    fig = create_interactive_pie_chart(
                                        suggestion['data'],
                                        suggestion['description']
                                    )
                                st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 🖼️ TAB 4: IMAGE CLASSIFICATION
# ============================================================

with tab4:
    st.markdown("### 🖼️ Fish Species Classification")
    st.markdown("Upload an image of a fish for AI-powered species identification.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image_file = st.file_uploader(
            "Upload Fish Image",
            type=['jpg', 'jpeg', 'png', 'gif', 'bmp'],
            key="image_classifier"
        )
        
        if image_file:
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 Classify Species", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing the image..."):
                    files = {'file': (image_file.name, image_file.getvalue(), image_file.type)}
                    result = make_api_request("/classify/image", method="POST", files=files)
                    
                    if result and result.get('status') == 'success':
                        with col2:
                            st.markdown("### 🎯 Classification Results")
                            
                            st.markdown(f"""
                            <div class="success-alert fade-in">
                                <h2>🐠 {result['predicted_species']}</h2>
                                <h3>Confidence: {result['confidence']*100:.1f}%</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Confidence meter
                            st.progress(result['confidence'])
                            
                            st.markdown("#### 🏆 Top 3 Predictions")
                            
                            for i, pred in enumerate(result.get('top3_predictions', []), 1):
                                confidence = pred['confidence'] * 100
                                st.markdown(f"""
                                <div class="info-card">
                                    <strong>{i}. {pred['species']}</strong><br>
                                    Confidence: {confidence:.1f}%
                                </div>
                                """, unsafe_allow_html=True)
                                st.progress(pred['confidence'])
                    
                    elif result and result.get('status') == 'no_model':
                        st.warning("⚠️ ML model not available. Please ensure the model is trained.")
                    else:
                        st.error("❌ Classification failed. Please try again.")

# ============================================================
# 📈 TAB 5: ANALYTICS & STATS
# ============================================================

with tab5:
    st.markdown("### 📈 System Analytics & Statistics")
    
    if stats_data:
        statistics = stats_data.get('statistics', {})
        
        # Session Info
        st.markdown("#### 🎯 Current Session")
        session_info = statistics.get('session_info', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents Processed", session_info.get('documents_processed', 0))
        with col2:
            st.metric("Queries Processed", session_info.get('queries_processed', 0))
        with col3:
            st.metric("Images Classified", session_info.get('images_classified', 0))
        
        # Database Stats
        st.markdown("#### 🗄️ Vector Database")
        db_stats = statistics.get('database_stats', {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Documents", db_stats.get('total_documents', 0))
        with col2:
            st.metric("Collection", db_stats.get('collection_name', 'N/A'))
        
        # File Processing Stats
        st.markdown("#### 📁 File Processing Statistics")
        file_stats = statistics.get('file_processing_stats', {})
        
        if file_stats:
            fig = create_interactive_bar_chart(
                file_stats,
                "Files Processed by Type",
                "File Type",
                "Count"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # ML Model Info
        st.markdown("#### 🤖 Machine Learning Model")
        
        col1, col2 = st.columns(2)
        with col1:
            ml_status = statistics.get('ml_model_status', 'unknown')
            status_color = "🟢" if ml_status == "loaded" else "🔴"
            st.markdown(f"**Status:** {status_color} {ml_status.upper()}")
        with col2:
            st.metric("Species Trained", statistics.get('ml_species_count', 0))
        
        # System Configuration
        with st.expander("⚙️ System Configuration"):
            config_data = make_api_request("/config")
            if config_data:
                st.json(config_data)
        
        # Refresh button
        if st.button("🔄 Refresh Statistics", use_container_width=True):
            st.rerun()

# ============================================================
# 🔗 SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### 🐠 MeenaSetu AI")
    st.markdown("---")
    
    st.markdown("#### 🎯 Quick Actions")
    
    if st.button("🏠 Home", use_container_width=True):
        st.rerun()
    
    if st.button("📚 Documentation", use_container_width=True):
        st.info("Documentation coming soon!")
    
    if st.button("❓ Help", use_container_width=True):
        st.info("""
        **How to use MeenaSetu AI:**
        
        1. **Chat:** Ask questions about fish species
        2. **Upload:** Add PDF, CSV, JSON, images
        3. **Visualize:** Create charts from data
        4. **Classify:** Identify fish from images
        5. **Analyze:** View system statistics
        """)
    
    st.markdown("---")
    st.markdown("#### 📊 Session Stats")
    st.metric("Queries", st.session_state.stats['queries'])
    st.metric("Uploads", st.session_state.stats['uploads'])
    st.metric("Visualizations", st.session_state.stats['visualizations'])
    
    st.markdown("---")
    st.markdown("#### ℹ️ About")
    st.markdown("""
    **Version:** 1.0.0  
    **Powered by:**  
    - 🤖 LangChain RAG
    - 🧠 Groq LLM
    - 📊 ML Classification
    - 🎨 Plotly Visualizations
    
    ---
    Made with ❤️ By Amrish Kumar Tiwary Lead AI Full Stack Developer""")