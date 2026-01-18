import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Optional, Dict, Any, List
import base64
from PIL import Image
import io

# PAGE CONFIGURATION
st.set_page_config(
    page_title="🐠 MeenaSetu AI",
    page_icon="🐠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-size: 3.5rem;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.2);
        font-weight: 700;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.95);
        font-size: 1.3rem;
        margin-top: 0.8rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.8rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }
    
    .metric-card h2 { margin: 0; font-size: 3rem; font-weight: 700; }
    .metric-card p { margin: 0.5rem 0 0 0; opacity: 0.95; font-size: 1.1rem; }
    
    .user-message {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #1E88E5;
    }
    
    .ai-message {
        background: linear-gradient(135deg, #F3E5F5 0%, #E1BEE7 100%);
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #9C27B0;
    }
    
    .classification-card {
        background: linear-gradient(135deg, #C8E6C9 0%, #A5D6A7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
    }
    
    .disease-alert {
        background: linear-gradient(135deg, #FFCCBC 0%, #FFAB91 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #FF5722;
    }
    
    .success-box {
        background: linear-gradient(135deg, #C8E6C9 0%, #A5D6A7 100%);
        color: #1B5E20;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
        font-weight: 600;
    }
    
    .error-box {
        background: linear-gradient(135deg, #FFCDD2 0%, #EF9A9A 100%);
        color: #B71C1C;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #F44336;
        margin: 1rem 0;
        font-weight: 600;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# CONFIGURATION
API_BASE_URL = "http://localhost:8000"

# SESSION STATE INITIALIZATION
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

if 'stats' not in st.session_state:
    st.session_state.stats = {'queries': 0, 'uploads': 0, 'visualizations': 0, 'classifications': 0}
# HELPER FUNCTIONS

def make_api_request(endpoint: str, method: str = "GET", **kwargs) -> Optional[Dict]:
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=30, **kwargs)
        elif method == "POST":
            response = requests.post(url, timeout=60, **kwargs)
        elif method == "DELETE":
            response = requests.delete(url, timeout=30, **kwargs)
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.ConnectionError:
        st.error(" Cannot connect to API server at http://localhost:8000")
        return None
    except requests.exceptions.Timeout:
        st.error(" Request timeout")
        return None
    except Exception as e:
        st.error(f" Error: {str(e)}")
        return None

def get_file_icon(file_type: str) -> str:
    """Get emoji for file type"""
    icons = {'pdf': '📄', 'csv': '📊', 'json': '📋', 'txt': '📝', 'jpg': '🖼️', 'jpeg': '🖼️', 'png': '🖼️'}
    return icons.get(file_type.lower().replace('.', ''), '')

def create_plotly_bar_chart(data: Dict, title: str, xlabel: str = "", ylabel: str = ""):
    """Create bar chart"""
    df = pd.DataFrame(list(data.items()), columns=['Category', 'Value'])
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['Category'],
            y=df['Value'],
            marker=dict(color=df['Value'], colorscale='Viridis'),
            text=df['Value'],
            texttemplate='%{text:.1f}',
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title=xlabel or 'Category',
        yaxis_title=ylabel or 'Value',
        template='plotly_white',
        height=500
    )
    return fig

def create_plotly_pie_chart(data: Dict, title: str):
    """Create pie chart"""
    fig = go.Figure(data=[
        go.Pie(
            labels=list(data.keys()),
            values=list(data.values()),
            hole=0.4,
            textinfo='label+percent'
        )
    ])
    
    fig.update_layout(title=title, template='plotly_white', height=500)
    return fig
# MAIN HEADER
st.markdown("""
<div class="main-header">
    <h1>🐠 MeenaSetu AI</h1>
    <p>Intelligent Aquatic Biodiversity Expert System</p>
    <p style="font-size: 1rem; margin-top: 0.8rem;">
        🤖 Multi-Model Ensemble | 📚 50K+ Documents | 🧠 Advanced RAG | Fish Disease and Species Detection
    </p>
</div>
""", unsafe_allow_html=True)
# HEALTH CHECK
health_data = make_api_request("/health")

if health_data and health_data.get('status') == 'healthy':
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2.5rem;"></div>
            <h2>{health_data['metrics']['vector_db_documents']:,}</h2>
            <p>Documents</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2.5rem;"></div>
            <h2>{health_data['metrics']['ml_models_loaded']}</h2>
            <p>ML Models</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2.5rem;"></div>
            <h2>{health_data['metrics']['queries_processed']}</h2>
            <p>Queries</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2.5rem;"></div>
            <h2>ONLINE</h2>
            <p>Status</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.error(" API server is not running. Start it with: `uvicorn main:app --reload`")
    st.stop()

st.markdown("<br>", unsafe_allow_html=True)
# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Chat",
    "🖼️ Fish ID",
    "🔬 Disease",
    "📊 Visualize",
    "📈 Stats"
])

# TAB 1: CHAT - COMPLETELY FIXED VERSION
with tab1:
    st.markdown("### 💬 Chat with MeenaSetu AI")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_area("Your Question:", placeholder="Ask about fish species, aquaculture, or upload an image...", height=100)
    
    with col2:
        st.markdown("** Optional Image**")
        chat_image = st.file_uploader("Upload fish image", type=['jpg', 'jpeg', 'png'], key="chat_img", label_visibility="collapsed")
        
        if chat_image:
            img = Image.open(chat_image)
            st.image(img, caption="Attached", width=200)
    
    col_a, col_b = st.columns([1, 5])
    
    with col_a:
        ask_btn = st.button(" Ask", type="primary")
    
    with col_b:
        clear_btn = st.button("🧹 Clear")
    
    if clear_btn:
        make_api_request("/conversation/clear", method="DELETE")
        st.session_state.conversation_history = []
        st.success(" Cleared!")
        time.sleep(1)
        st.rerun()
    
    if ask_btn and query:
        with st.spinner("🤔 Thinking..."):
            classify_result = None
            disease_result = None
            if chat_image:
                chat_image.seek(0)
                
                # Species Classification
                st.info("🐟 Classifying species...")
                files_classify = {'file': (chat_image.name, chat_image.getvalue(), chat_image.type)}
                classify_result = make_api_request(
                    "/classify/fish", 
                    method="POST", 
                    files=files_classify,
                    data={'detect_disease': 'false'}
                )
                
                # Disease Detection
                st.info("🔬 Checking for diseases...")
                chat_image.seek(0)
                files_disease = {'file': (chat_image.name, chat_image.getvalue(), chat_image.type)}
                disease_result = make_api_request(
                    "/detect/disease", 
                    method="POST", 
                    files=files_disease,
                    data={'description': query, 'get_treatment': 'true'}
                )
            context_parts = []
            
            # Add species info
            if classify_result and classify_result.get('status') == 'success':
                species = classify_result.get('species', 'Unknown')
                conf = classify_result.get('confidence', 0)
                context_parts.append(f"Species Identified: {species} (Confidence: {conf*100:.1f}%)")
            
            # Add disease info - CRITICAL FIX: Only send clean summary
            if disease_result:
                api_status = disease_result.get('status', '')
                
                if api_status == 'healthy':
                    # Fish is healthy
                    context_parts.append(
                        "Health Status: The fish appears HEALTHY. "
                        "No significant diseases were detected by the AI model."
                    )
                    
                    # Add note about model uncertainty if present
                    if disease_result.get('warning'):
                        context_parts.append(
                            f"Note: {disease_result.get('warning')}"
                        )
                
                elif api_status == 'detected':
                    # Disease detected
                    disease_name = disease_result.get('primary_disease', 'Unknown Disease')
                    confidence = disease_result.get('confidence', 0)
                    
                    context_parts.append(
                        f"Health Status: DISEASE DETECTED - {disease_name} "
                        f"(AI Confidence: {confidence*100:.1f}%)"
                    )
                    
                    # Add treatment recommendations if available
                    if disease_result.get('recommendations'):
                        recs_text = "\n".join([f"- {rec}" for rec in disease_result['recommendations'][:3]])
                        context_parts.append(f"Recommended Actions:\n{recs_text}")
                
                elif api_status == 'error':
                    context_parts.append(
                        f"Health Status: Unable to detect diseases - {disease_result.get('message', 'Error occurred')}"
                    )
            
            # ========================================
            # STEP 3: Create final prompt for LLM
            # ========================================
            if context_parts:
                # Build structured context
                structured_context = "\n\n".join(context_parts)
                
                # Create enhanced query
                enhanced_query = f"""Based on the following fish analysis results:

{structured_context}

User's Question: {query}

Please provide a helpful answer in a friendly, conversational tone. Mix Hindi and English naturally. Use emojis appropriately."""
            else:
                # No image - just use original query
                enhanced_query = query
            
            # ========================================
            # STEP 4: Get AI response
            # ========================================
            st.info("🤖 Generating answer...")
            result = make_api_request("/query/simple", method="POST", data={"query": enhanced_query})
            
            if result:
                # Store in history with CLEAN data
                st.session_state.conversation_history.append({
                    'query': query,
                    'response': result,
                    'classification': classify_result,
                    'disease': disease_result,
                    'timestamp': datetime.now().isoformat()
                })
                st.session_state.stats['queries'] += 1
                
                if classify_result:
                    st.session_state.stats['classifications'] += 1
                
                st.rerun()

    if st.session_state.conversation_history:
        st.markdown("---")
        st.markdown("### 📜 Conversation")
        
        for conv in reversed(st.session_state.conversation_history[-5:]):
            # User message
            st.markdown(
                f'<div class="user-message"><strong>🙋 You:</strong><br>{conv["query"]}</div>', 
                unsafe_allow_html=True
            )
            
            # Species classification card
            if conv.get('classification') and conv['classification'].get('status') == 'success':
                cls = conv['classification']
                st.markdown(f"""
                <div class="classification-card">
                    <strong> Species Detected:</strong> <strong>{cls['species']}</strong> 
                    (Confidence: {cls['confidence']*100:.1f}%)
                </div>
                """, unsafe_allow_html=True)
                st.progress(cls['confidence'])
            
            # Disease detection card - PROPERLY FIXED
            if conv.get('disease'):
                disease = conv['disease']
                api_status = disease.get('status', '')
                
                if api_status == 'detected':
                    # Disease detected - show alert
                    disease_name = disease.get('primary_disease', 'Unknown')
                    disease_conf = disease.get('confidence', 0)
                    
                    st.markdown(f"""
                    <div class="disease-alert">
                        <h4> Disease Detected: {disease_name}</h4>
                        <p><strong>Confidence:</strong> {disease_conf*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.progress(disease_conf)
                    
                    # Show recommendations
                    if disease.get('recommendations'):
                        with st.expander(" View Treatment Recommendations", expanded=True):
                            for i, rec in enumerate(disease['recommendations'], 1):
                                st.markdown(f"**{i}.** {rec}")
                
                elif api_status == 'healthy':
                    # Healthy - show success message
                    st.markdown("""
                    <div class="success-box">
                         <strong>Fish appears healthy!</strong> No significant diseases detected.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show warning if model was uncertain
                    if disease.get('warning'):
                        with st.expander("ℹ Technical Details"):
                            st.info(disease['warning'])
                            st.caption(f"Model confidence: {disease.get('confidence', 0)*100:.1f}%")
                
                elif api_status == 'error':
                    st.markdown(f"""
                    <div class="error-box">
                         Disease detection error: {disease.get('message', 'Unknown error')}
                    </div>
                    """, unsafe_allow_html=True)
                
                elif api_status == 'unavailable':
                    st.warning(" Disease detection is currently unavailable")
            
            # AI response
            st.markdown(
                f'<div class="ai-message"><strong> MeenaSetu:</strong><br>{conv["response"]["answer"]}</div>', 
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)
# TAB 2: FISH CLASSIFICATION
with tab2:
    st.markdown("### 🖼️ Fish Species Identification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fish_image = st.file_uploader("Upload Fish Image", type=['jpg', 'jpeg', 'png'], key="fish_id")
        
        if fish_image:
            img = Image.open(fish_image)
            st.image(img, caption="Fish Image", width=400)
            
            if st.button("🔍 Identify Species", type="primary"):
                with st.spinner("🤖 Analyzing..."):
                    files = {'file': (fish_image.name, fish_image.getvalue(), fish_image.type)}
                    result = make_api_request("/classify/fish", method="POST", files=files)
                    
                    if result and result.get('status') == 'success':
                        with col2:
                            st.markdown("### 🎯 Results")
                            
                            st.markdown(f"""
                            <div class="classification-card">
                                <h2>🐠 {result['species']}</h2>
                                <p style="font-size: 1.5rem;"><strong>Confidence: {result['confidence']*100:.2f}%</strong></p>
                                <p><strong>Model:</strong> {result['model_info']['model_used']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.progress(result['confidence'])
                            
                            st.markdown("#### 🏆 Top 3 Predictions")
                            for i, pred in enumerate(result['top3_predictions'], 1):
                                st.write(f"**{i}. {pred['species']}** - {pred['confidence']*100:.2f}%")
                                st.progress(pred['confidence'])
                            
                            st.session_state.stats['classifications'] += 1

# ============================================================
# TAB 3: DISEASE DETECTION
# ============================================================
with tab3:
    st.markdown("### 🔬 Fish Disease Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        disease_image = st.file_uploader("Upload Sick Fish Image", type=['jpg', 'jpeg', 'png'], key="disease_img")
        
        if disease_image:
            img = Image.open(disease_image)
            st.image(img, caption="Disease Analysis", width=400)
            
            description = st.text_area("Describe symptoms (optional):", height=80)
            
            if st.button("🔬 Detect Disease", type="primary"):
                with st.spinner("🔬 Analyzing..."):
                    files = {'file': (disease_image.name, disease_image.getvalue(), disease_image.type)}
                    data = {'description': description, 'get_treatment': 'true'}
                    
                    result = make_api_request("/detect/disease", method="POST", files=files, data=data)
                    
                    if result:
                        with col2:
                            if result.get('status') == 'detected':
                                st.markdown(f"""
                                <div class="disease-alert">
                                    <h3>⚠️ {result['primary_disease']}</h3>
                                    <p><strong>Confidence:</strong> {result.get('confidence', 0)*100:.1f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                if result.get('recommendations'):
                                    st.markdown("#### 💊 Treatment")
                                    for i, rec in enumerate(result['recommendations'], 1):
                                        st.write(f"**{i}.** {rec}")
                            
                            elif result.get('status') == 'healthy':
                                st.markdown('<div class="success-box">✅ No disease detected!</div>', unsafe_allow_html=True)
                            
                            else:
                                st.markdown(f'<div class="error-box">❌ {result.get("message", "Detection failed")}</div>', unsafe_allow_html=True)

# ============================================================
# TAB 4: VISUALIZATIONS
# ============================================================
with tab4:
    st.markdown("### 📊 Data Visualizations")
    
    viz_mode = st.radio("Mode:", ["Manual Entry", "Upload CSV"], horizontal=True)
    
    if viz_mode == "Manual Entry":
        col1, col2 = st.columns(2)
        
        with col1:
            chart_type = st.selectbox("Chart Type", ["Bar", "Pie", "Line"])
            title = st.text_input("Title", "My Chart")
            num_points = st.slider("Data points", 2, 10, 5)
        
        with col2:
            if chart_type in ["Bar", "Line"]:
                xlabel = st.text_input("X-axis", "Category")
                ylabel = st.text_input("Y-axis", "Value")
            else:
                xlabel = ylabel = ""
        
        st.markdown("#### Enter Data")
        data = {}
        
        for i in range(num_points):
            col_l, col_v = st.columns(2)
            with col_l:
                label = st.text_input(f"Label {i+1}", f"Item {i+1}", key=f"l_{i}")
            with col_v:
                value = st.number_input(f"Value {i+1}", 0.0, 10000.0, float((i+1)*10), key=f"v_{i}")
            data[label] = value
        
        if st.button("🎨 Generate", type="primary"):
            with st.spinner("Creating..."):
                if chart_type == "Bar":
                    fig = create_plotly_bar_chart(data, title, xlabel, ylabel)
                elif chart_type == "Pie":
                    fig = create_plotly_pie_chart(data, title)
                else:  # Line
                    df = pd.DataFrame(list(data.items()), columns=['X', 'Y'])
                    fig = px.line(df, x='X', y='Y', title=title, markers=True)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Save via API
                api_type = chart_type.lower()
                api_data = {
                    "plot_type": api_type,
                    "data": data,
                    "title": title,
                    "xlabel": xlabel,
                    "ylabel": ylabel
                }
                
                result = make_api_request("/visualize/create", method="POST", json=api_data)
                
                if result and result.get('status') == 'success':
                    st.success(f"✅ Saved! Download: {API_BASE_URL}{result.get('download_url', '')}")
                    st.session_state.stats['visualizations'] += 1
    
    else:  # CSV Upload
        csv_file = st.file_uploader("Upload CSV", type=['csv'], key="csv_viz")
        
        if csv_file:
            df = pd.read_csv(csv_file)
            st.dataframe(df.head(10))
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_col = st.selectbox("X-axis", categorical_cols if categorical_cols else df.columns.tolist())
            
            with col2:
                y_col = st.selectbox("Y-axis", numeric_cols if numeric_cols else df.columns.tolist())
            
            with col3:
                auto_type = st.selectbox("Type", ["Bar", "Pie", "Line", "Scatter"])
            
            if st.button("📊 Create", type="primary"):
                with st.spinner("Generating..."):
                    try:
                        if auto_type == "Bar" and categorical_cols and numeric_cols:
                            data = df.groupby(x_col)[y_col].mean().to_dict()
                            fig = create_plotly_bar_chart(data, f"{y_col} by {x_col}", x_col, y_col)
                        
                        elif auto_type == "Pie" and x_col in categorical_cols:
                            data = df[x_col].value_counts().head(10).to_dict()
                            fig = create_plotly_pie_chart(data, f"Distribution of {x_col}")
                        
                        elif auto_type == "Scatter":
                            fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                        
                        else:
                            df_sorted = df.sort_values(x_col)
                            fig = px.line(df_sorted, x=x_col, y=y_col, title=f"{y_col} vs {x_col}", markers=True)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        st.success("✅ Visualization created!")
                        st.session_state.stats['visualizations'] += 1
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

# ============================================================
# TAB 5: ANALYTICS
# ============================================================
with tab5:
    st.markdown("### 📈 System Statistics")
    
    stats_data = make_api_request("/stats")
    
    if stats_data:
        statistics = stats_data.get('statistics', {})
        session_info = statistics.get('session_info', {})
        db_stats = statistics.get('database_stats', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Docs Processed", session_info.get('documents_processed', 0))
        
        with col2:
            st.metric("Queries", session_info.get('queries_processed', 0))
        
        with col3:
            st.metric("Images", session_info.get('images_classified', 0))
        
        with col4:
            st.metric("Viz Created", st.session_state.stats['visualizations'])
        
        st.markdown("---")
        st.markdown("#### 🗄️ Vector Database")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Documents", db_stats.get('total_documents', 0))
        
        with col2:
            st.metric("ML Status", statistics.get('ml_model_status', 'unknown').upper())
        
        # Species list
        species_data = make_api_request("/docs/species-list")
        
        if species_data and species_data.get('species'):
            st.markdown("---")
            st.markdown(f"#### 🐠 Classifiable Species ({species_data.get('total_species', 0)})")
            
            with st.expander("View All Species"):
                species_list = species_data.get('species', [])
                cols = st.columns(3)
                
                for i, species in enumerate(species_list):
                    with cols[i % 3]:
                        st.write(f"• {species}")
        
        if st.button("🔄 Refresh"):
            st.rerun()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🐠 MeenaSetu AI")
    st.markdown("---")
    
    st.markdown("#### 📊 Session Stats")
    st.write(f"- Queries: {st.session_state.stats['queries']}")
    st.write(f"- Classifications: {st.session_state.stats['classifications']}")
    st.write(f"- Visualizations: {st.session_state.stats['visualizations']}")
    
    st.markdown("---")
    st.markdown("#### 🚀 Quick Actions")
    
    if st.button("📚 API Docs"):
        st.write(f"[Open Docs]({API_BASE_URL}/docs)")
    
    if st.button("💾 Export Chat"):
        st.write(f"[Download]({API_BASE_URL}/conversation/export)")
    
    st.markdown("---")
    st.markdown("#### ℹ️ About")
    st.markdown("""
    **MeenaSetu AI v2.0**
    
    - 🤖 3 ML Models
    - 📚 50K+ Documents
    - 💬 Groq LLM
    - 🔬 Disease Detection
    - Species detection
    - Data Visualizations
    
    Made with ❤️ by  
    **Amrish Kumar Tiwary**
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🐠 MeenaSetu AI - Intelligent Aquatic Expert | Version 2.0.0 | © 2026</p>
</div>
""", unsafe_allow_html=True)