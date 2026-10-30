# -Hybrid-Semantic-Search-and-Document-Retrieval-System
Hybrid search engine using BM25 keyword matching and semantic similarity with Sentence Transformers and FAISS

## 📌 Project Overview

The Hybrid Semantic Search Engine is a Python-based application that retrieves relevant documents by combining keyword-based search and semantic search techniques.
It uses BM25 for keyword matching and Sentence Transformers for generating text embeddings. FAISS is used to perform efficient similarity searches and rank relevant documents.
The application provides an interactive interface using Streamlit, allowing users to enter queries and explore relevant search results.

## 🎯 Objectives

- Retrieve relevant documents based on user queries.
- Combine keyword matching and semantic similarity.
- Improve document retrieval using hybrid ranking.
- Provide a simple and interactive search interface.

## 🚀 Features

- Hybrid search using BM25 and semantic similarity.
- Text embedding generation using Sentence Transformers.
- Efficient vector similarity search using FAISS.
- Adjustable keyword search weight.
- Display ranked search results.
- Support for sample documents and CSV document uploads.
- Interactive interface using Streamlit.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **NLP:** Sentence Transformers
- **Keyword Search:** BM25
- **Vector Search:** FAISS
- **Data Processing:** NumPy, Pandas
- **Frontend:** Streamlit

## 📂 Project Structure

hybrid_semantic_search/
│
├── app.py
├── requirements.txt
├── README.me
└── .venv/

Note: The `.venv` folder should not be uploaded to GitHub.

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/deepthi-bolloju/-Hybrid-Semantic-Search-and-Document-Retrieval-System.git
```

### 2. Navigate to the Project Folder

```bash
cd Hybrid-Semantic-Search-and-Document-Retrieval-System.git
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Run the Application

```bash
python -m streamlit run app.py
```

The application will open in your browser at:

http://localhost:8501

## 🔍 How It Works

1. Load sample documents or upload a CSV file.
2. Enter a search query.
3. BM25 calculates keyword relevance scores.
4. Sentence Transformers generate embeddings for semantic comparison.
5. FAISS retrieves semantically similar documents.
6. The system combines the scores using weighted hybrid ranking.
7. The most relevant documents are displayed in ranked order.

## 💡 Example Search Queries

- Machine Learning
- Artificial Intelligence
- Natural Language Processing
- How do computers learn from data?
- Deep Learning Applications

## 🌍 Applications

- Enterprise document search
- Knowledge management systems
- Educational content retrieval
- Technical documentation search
- Frequently Asked Questions (FAQ) retrieval

## 🔮 Future Enhancements

- Support for PDF and DOCX documents.
- Integration with Retrieval-Augmented Generation (RAG).
- Larger document collections.
- Search performance evaluation using Precision@K and Recall@K.
- Improved ranking and filtering techniques.

## 👩‍💻 Author

**Deepthi Bolloju**
