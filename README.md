# 🛫 SkyAnalyst-AI: Cabin Operations Intelligence Framework

**SkyAnalyst-AI** is a modular NLP framework designed to process, analyze, and summarize flight cabin reports. Leveraging **RAG (Retrieval-Augmented Generation)** and **Abstractive Summarization**, it transforms unstructured crew narratives into structured operational insights.

> Built as a portfolio project targeting aviation data analytics roles — specifically designed around real-world cabin operations workflows.

---

## 🎯 Project Significance

Aviation safety depends on identifying patterns in thousands of daily cabin reports. This project provides:

* **Semantic Retrieval:** Meaning-based search across 2,500+ incident reports using Vector Embeddings — finds "kitchen equipment failure" even when the report says "galley oven unserviceable."
* **Automated Insights:** Abstractive summarization of technical, medical, and safety incidents using DistilBART-CNN.
* **Operational Analytics:** Statistical mapping of flight duration vs. incident intensity across 5 report categories.

---

## 🧠 Technical Architecture

The application follows a dual-stream architecture:

### 1. RAG Pipeline (Retrieval-Augmented Generation)

* **Vector Engine:** ChromaDB for local, persistent high-dimensional data storage.
* **Embeddings:** `all-MiniLM-L6-v2` (384-dimensional dense vectors).
* **Inference:** Direct model inference using **DistilBART-CNN** for high-speed executive summaries.

### 2. Operational EDA (Exploratory Data Analysis)

* **Sentiment Mapping:** Real-time sentiment intensity calculation via TextBlob.
* **Statistical Viz:** Seaborn-based Boxplots and Violin plots for duration-category correlation.

---

## 📈 Evaluation & Optimization

This project was optimized for low-latency inference and semantic precision:

| Feature | Implementation | Engineering Benefit |
| --- | --- | --- |
| **Search** | Semantic Similarity | Bypasses keyword limitations (e.g., Galley vs. Kitchen) |
| **Summary** | DistilBART (Compressed) | 4x faster than standard BART with negligible quality loss |
| **Storage** | ChromaDB (Vector DB) | Scalable retrieval for large-scale aviation datasets |
| **Data** | Synthetic generation via Faker | 2,500 realistic reports across 22 flight routes and 5 incident categories |

The summarization pipeline builds on prior NLP research achieving **0.52 ROUGE-L** on Turkish low-resource datasets.

---

## 🗂️ Project Structure

```bash
├── src/
│   ├── app.py                  # Main Multi-tab Streamlit Application
│   ├── build_vector_db.py      # Builds ChromaDB Vector Database
│   ├── generate_data.py        # Generates synthetic cabin reports
│   └── templates.json          # Report templates by incident category
├── data/
│   ├── cabin_reports.csv       # Generated operational dataset (not tracked)
│   └── cabin_vector_db/        # Persistent ChromaDB collection (not tracked)
├── notebooks/
│   └── data_exploration.ipynb  # Initial EDA and RAG testing
└── requirements.txt            # Project dependencies
```

---

## 🚀 Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic dataset
python src/generate_data.py

# 3. Build vector database
python src/build_vector_db.py

# 4. Launch the dashboard
streamlit run src/app.py
```

---

## 🛠️ Tech Stack

`Python` `Streamlit` `ChromaDB` `Sentence-Transformers` `HuggingFace Transformers` `DistilBART` `TextBlob` `Pandas` `Seaborn` `Faker`