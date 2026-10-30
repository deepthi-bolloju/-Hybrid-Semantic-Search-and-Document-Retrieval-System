import streamlit as st
import pandas as pd
import numpy as np
import faiss

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# --------------------------------
# 1. Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Hybrid Semantic Search Engine",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Hybrid Semantic Search Engine")

st.write(
    "Search documents using keyword matching "
    "and semantic similarity."
)


# --------------------------------
# 2. Sample Dataset
# --------------------------------

DEFAULT_DOCUMENTS = [
    "Machine learning enables computers to learn from data.",
    "Deep learning uses neural networks to solve complex problems.",
    "Natural language processing helps computers understand human language.",
    "Python is a popular programming language for data science.",
    "Artificial intelligence allows machines to perform intelligent tasks.",
    "Data science involves statistics, programming, and data analysis.",
    "Semantic search finds information based on meaning and context.",
    "Information retrieval helps users find relevant documents.",
    "Generative AI can generate text, images, and other content.",
    "Computer vision enables computers to understand images and videos.",
    "Supervised learning uses labeled data to train machine learning models.",
    "Unsupervised learning discovers patterns in unlabeled data.",
    "Neural networks are inspired by the structure of the human brain.",
    "Text classification assigns categories to text documents.",
    "Search engines help users find useful information quickly."
]


# --------------------------------
# 3. Load Sentence Transformer
# --------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


with st.spinner("Loading the semantic search model..."):
    model = load_model()


# --------------------------------
# 4. Select Dataset
# --------------------------------

st.sidebar.header("Dataset Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file (optional)",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        uploaded_df = pd.read_csv(uploaded_file)

        if "document" not in uploaded_df.columns:
            st.error(
                "Your CSV must contain a column named 'document'."
            )
            st.stop()

        documents = (
            uploaded_df["document"]
            .dropna()
            .astype(str)
            .tolist()
        )

        documents = [
            doc.strip()
            for doc in documents
            if doc.strip()
        ]

        if not documents:
            st.error("The CSV contains no usable documents.")
            st.stop()

    except Exception as error:
        st.error(f"Could not read the CSV: {error}")
        st.stop()

else:
    documents = DEFAULT_DOCUMENTS


st.sidebar.success(
    f"{len(documents)} documents loaded"
)


# --------------------------------
# 5. Build BM25 Keyword Search
# --------------------------------

tokenized_documents = [
    doc.lower().split()
    for doc in documents
]

bm25 = BM25Okapi(tokenized_documents)


# --------------------------------
# 6. Build Semantic Search
# --------------------------------

with st.spinner("Preparing document embeddings..."):

    document_embeddings = model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    document_embeddings = document_embeddings.astype(
        "float32"
    )

    dimension = document_embeddings.shape[1]

    faiss_index = faiss.IndexFlatIP(dimension)

    faiss_index.add(document_embeddings)


# --------------------------------
# 7. Hybrid Search Function
# --------------------------------

def hybrid_search(
    query,
    top_k=5,
    alpha=0.5
):

    # Keyword-based search using BM25
    query_tokens = query.lower().split()

    keyword_scores = np.array(
        bm25.get_scores(query_tokens),
        dtype="float32"
    )

    # Normalize keyword scores
    max_keyword_score = keyword_scores.max()

    if max_keyword_score > 0:
        keyword_scores = (
            keyword_scores / max_keyword_score
        )

    # Semantic search using Sentence Transformers
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    semantic_scores_raw, semantic_indices = (
        faiss_index.search(
            query_embedding,
            len(documents)
        )
    )

    # Store semantic scores in document order
    semantic_scores = np.zeros(
        len(documents),
        dtype="float32"
    )

    for score, index_position in zip(
        semantic_scores_raw[0],
        semantic_indices[0]
    ):

        if index_position >= 0:
            semantic_scores[index_position] = score

    # Convert cosine similarity to a 0-1 range
    semantic_scores = np.clip(
        (semantic_scores + 1) / 2,
        0,
        1
    )

    # Combine keyword and semantic scores
    hybrid_scores = (
        alpha * keyword_scores
        + (1 - alpha) * semantic_scores
    )

    # Rank documents by hybrid score
    ranked_indices = np.argsort(
        hybrid_scores
    )[::-1][:top_k]

    results = []

    for rank, index_position in enumerate(
        ranked_indices,
        start=1
    ):

        results.append({
            "Rank": rank,
            "Document": documents[index_position],
            "Keyword Score": round(
                float(keyword_scores[index_position]), 4
            ),
            "Semantic Score": round(
                float(semantic_scores[index_position]), 4
            ),
            "Hybrid Score": round(
                float(hybrid_scores[index_position]), 4
            )
        })

    return pd.DataFrame(results)


# --------------------------------
# 8. Search Interface
# --------------------------------

st.subheader("Search Documents")

query = st.text_input(
    "Enter your search query",
    placeholder="Example: How can computers understand language?"
)

alpha = st.slider(
    "Keyword Search Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1
)

top_k = st.slider(
    "Number of Results",
    min_value=1,
    max_value=min(10, len(documents)),
    value=min(5, len(documents))
)


# --------------------------------
# 9. Display Results
# --------------------------------

if query.strip():

    with st.spinner("Searching documents..."):

        results = hybrid_search(
            query=query,
            top_k=top_k,
            alpha=alpha
        )

    st.subheader("Search Results")

    st.caption(
        f"Keyword weight: {alpha:.1f} | "
        f"Semantic weight: {1 - alpha:.1f}"
    )

    if not results.empty:

        for _, row in results.iterrows():

            st.markdown(
                f"### {int(row['Rank'])}. "
                f"{row['Document']}"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Keyword Score",
                f"{row['Keyword Score']:.4f}"
            )

            col2.metric(
                "Semantic Score",
                f"{row['Semantic Score']:.4f}"
            )

            col3.metric(
                "Hybrid Score",
                f"{row['Hybrid Score']:.4f}"
            )

            st.divider()

        # Download search results
        csv_data = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Search Results",
            data=csv_data,
            file_name="search_results.csv",
            mime="text/csv"
        )

else:

    st.info(
        "Enter a query above to find relevant documents."
    )


# --------------------------------
# 10. Display Dataset
# --------------------------------

with st.expander("View All Documents"):

    dataset_df = pd.DataFrame({
        "Document": documents
    })

    st.dataframe(
        dataset_df,
        use_container_width=True
    )