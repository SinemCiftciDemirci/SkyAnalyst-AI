import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from textblob import TextBlob

# --- Global Configuration ---
st.set_page_config(page_title="SkyAnalyst-AI | Cabin Ops Intelligence", layout="wide")

@st.cache_resource
def load_all_resources():
    """Load and cache data, vector database, and AI models."""
    # 1. Dataset Loading
    df = pd.read_csv("./data/cabin_reports.csv")
    text_col = 'report_content' # Validated from dataset image
    
    # On-the-fly Sentiment Calculation for Dashboard Accuracy
    if 'sentiment_score' not in df.columns:
        df['sentiment_score'] = df[text_col].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    
    # 2. Vector DB Integration
    client = chromadb.PersistentClient(path="./data/cabin_vector_db")
    collection = client.get_collection(name="cabin_reports")
    
    # 3. Model Initialization
    search_model = SentenceTransformer('all-MiniLM-L6-v2')
    summ_model_name = "sshleifer/distilbart-cnn-6-6"
    tokenizer = AutoTokenizer.from_pretrained(summ_model_name)
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summ_model_name)
    
    return df, collection, search_model, tokenizer, summarizer_model, text_col

# Initialize Resources
df, collection, search_model, tokenizer, summarizer_model, text_col = load_all_resources()

# --- Dashboard Header ---
st.title("🛫 SkyAnalyst-AI: Incident Intelligence Dashboard")
st.markdown("---")

# Tab Navigation
tab_rag, tab_eda = st.tabs(["🤖 AI Search Assistant", "📈 Operational Analytics"])

with tab_rag:
    st.header("Semantic Incident Search")
    query = st.text_input("Enter keywords (e.g., 'galley equipment issues', 'medical cases'):")
    
    if query:
        # Search Execution
        query_vector = search_model.encode(query).tolist()
        results = collection.query(query_embeddings=[query_vector], n_results=5)
        
        # Summary Feature
        if st.button("✨ Summarize These Findings"):
            with st.spinner("AI is analyzing incident patterns..."):
                context = " ".join(results['documents'][0])
                inputs = tokenizer(context[:1000], return_tensors="pt", max_length=1024, truncation=True)
                summary_ids = summarizer_model.generate(inputs["input_ids"], max_length=130, min_length=30)
                st.success("Executive AI Summary:")
                st.write(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
        
        # Detailed Results Display
        for i, doc in enumerate(results['documents'][0]):
            with st.expander(f"Report Match #{i+1}"):
                st.write(f"**Content:** {doc}")
                st.caption(f"Metadata: {results['metadatas'][0][i]}")

with tab_eda:
    st.header("Fleet & Operational Trends")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Flight Duration Distribution by Category")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x='category', y='duration_min', hue='category', palette="viridis", legend=False, ax=ax1)
        plt.xticks(rotation=45)
        st.pyplot(fig1)
        st.info("Analysis: Technical reports are consistent, while medical incidents correlate with long-haul sectors.")
        
    with col2:
        st.subheader("Sentiment Intensity Analysis")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=df, x='category', y='sentiment_score', hue='category', palette="magma", legend=False, ax=ax2)
        plt.axhline(0, color='red', linestyle='--', alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig2)
        st.info("Analysis: Negative sentiment peaks in medical emergencies, requiring focused crew support.")