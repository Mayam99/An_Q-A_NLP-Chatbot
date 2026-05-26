import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & LAYOUT
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Chatbot | Question-Answering System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling to enhance UI appearance
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTextInput>div>div>input { background-color: #ffffff; }
    .context-box { 
        background-color: #e9ecef; 
        padding: 15px; 
        border-radius: 5px; 
        border-left: 5px solid #007bff;
        margin-bottom: 10px;
    }
    .answer-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CACHED MODEL RESOURCE INITIALIZATION
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Compiling RAG Architecture & Loading Weights...")
def initialize_rag_system():
    """Initializes the Embedding Model and Downstream Fine-tuned Extractive BERT Pipeline."""
    # Initialize the Sentence Transformer model matching your training setup
    embedding_model = SentenceTransformer("all-mpnet-base-v2")
    
    # Initialize your Fine-Tuned BERT Pipeline. 
    # NOTE: Update 'bert-base-uncased' to your local/huggingface checkpoint path if needed.
    qa_pipeline = pipeline(
        "question-answering",
        model="bert-base-uncased",
        tokenizer="bert-base-uncased"
    )
    return embedding_model, qa_pipeline

# Safe execution initialization
try:
    embed_model, bert_pipeline = initialize_rag_system()
except Exception as e:
    st.error(f"Failed to instantiate models: {str(e)}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. CORE PROCESSING LOGIC & DATA INGESTION
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_index_knowledge_base(data_path):
    """Loads text chunks, converts them to matrix arrays, and constructs flat FAISS index."""
    if not os.path.exists(data_path):
        # Create mock data structure if file is absent to prevent immediate breaking
        df = pd.DataFrame({
            'source': ['System Default Docs'],
            'chunk_id': [0],
            'content': ["Capital Adequacy Ratio is a measure of a bank's capital. It is expressed as a percentage of a bank's risk-weighted credit exposures."]
        })
    else:
        # Load preprocessed CSV (ensure columns: 'source', 'chunk_id', 'content' exist)
        df = pd.read_csv(data_path)
    
    # Compute base embeddings dynamically across data array
    # If chunks are immutable, precompute and store embeddings directly into your CSV to speed up boot times.
    with st.spinner("Generating document matrix vectorizations..."):
        embeddings = embed_model.encode(df['content'].tolist(), show_progress_bar=False)
        embeddings_matrix = np.array(embeddings).astype('float32')
    
    dimension = embeddings_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_matrix)
    
    return df, index

# Instantiate knowledge base
DATA_FILE_PATH = os.path.join("data", "documents.csv")
chunk_df, faiss_index = load_and_index_knowledge_base(DATA_FILE_PATH)

def execute_rag_pipeline(query, k=3):
    """Executes full retrieval over indexed arrays and feeds top context into BERT."""
    # 1. Vector Search across FAISS index
    query_vector = embed_model.encode([query]).astype('float32')
    distances, indices = faiss_index.search(query_vector, k)
    
    # 2. Extract context structures
    matched_rows = chunk_df.iloc[indices[0]]
    combined_context = " ".join(matched_rows['content'].tolist())
    
    # 3. Downstream BERT Extraction
    prediction = bert_pipeline(question=query, context=combined_context)
    
    return prediction, matched_rows

# -----------------------------------------------------------------------------
# 4. STREAMLIT USER INTERFACE (UI)
# -----------------------------------------------------------------------------
st.title("🤖 Intelligently Retained NLP Question-Answering Bot")
st.markdown("Interact seamlessly with your document warehouse using vector retrieval and Extractive BERT architectures.")

# Sidebar Configuration Control Panel
with st.sidebar:
    st.header("🔧 Configuration Control")
    retrieval_k = st.slider("Document Extraction Threshold (k)", min_value=1, max_value=5, value=3)
    
    st.markdown("---")
    st.subheader("Data Registry Metadata")
    st.write(f"Total Registered Knowledge Chunks: `{len(chunk_df)}`")
    st.write(f"FAISS Index Status: `READY (FlatL2 Dimension: {faiss_index.d})`")

# Main Interface Workspace Layout
user_query = st.text_input("Enter your query details below:", placeholder="What is capital adequacy ratio?")

if user_query.strip():
    with st.spinner("Executing dense context lookup and extraction..."):
        try:
            # Run the complete RAG logic pipeline
            answer_payload, source_documents = execute_rag_pipeline(user_query, k=retrieval_k)
            
            # Display Extracted Answer Profile
            st.subheader("💡 Extracted Model Answer")
            st.markdown(f"""
                <div class="answer-box">
                    <strong>Answer:</strong> {answer_payload['answer']}<br>
                    <small style='color: #6c757d;'>Confidence Score: {answer_payload['score']:.4f} | Character Segment Range: ({answer_payload['start']} - {answer_payload['end']})</small>
                </div>
            """, unsafe_allow_html=True)
            
            # Display Context Retained reference points
            st.subheader("📚 Extracted Context Reference Bundles")
            for idx, row in source_documents.iterrows():
                with st.expander(f"Source Document: {row['source']} (Chunk Index ID: {row['chunk_id']})"):
                    st.markdown(f"""
                        <div class="context-box">
                            {row['content']}
                        </div>
                    """, unsafe_allow_html=True)
                    
        except Exception as error:
            st.error(f"An anomaly occurred while running evaluation operations: {str(error)}")
else:
    st.info("System awaiting query parameters. Input your objective above to trigger inference operations.")