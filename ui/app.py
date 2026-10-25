import json
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st
import numpy as np
from src.config import (
    PDF_DIR,
    REGISTRY_PATH,
    GROQ_MODEL,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
)

from src.retriever import HybridRetriever
from src.generator import generate_answer


# ============================================================
# PAGE CONFIG
# ============================================================

METRICS_FILE = Path("storage/query_metrics.json")
METRICS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

def save_query_metrics(
    retrieval_ms,
    generation_ms,
    total_ms,
):
    metrics = []

    if METRICS_FILE.exists():
        try:
            metrics = json.loads(
                METRICS_FILE.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            metrics = []

    metrics.append(
        {
            "timestamp": time.time(),
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": total_ms,
        }
    )

    METRICS_FILE.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

def calculate_latency_percentiles():

    if not METRICS_FILE.exists():
        return None

    metrics = json.loads(
        METRICS_FILE.read_text(
            encoding="utf-8"
        )
    )

    if len(metrics) < 2:
        return None

    retrieval = [
        x["retrieval_ms"]
        for x in metrics
    ]

    total = [
        x["total_ms"]
        for x in metrics
    ]

    return {
        "count": len(metrics),

        "retrieval_p50": np.percentile(
            retrieval,
            50,
        ),

        "retrieval_p90": np.percentile(
            retrieval,
            90,
        ),

        "retrieval_p95": np.percentile(
            retrieval,
            95,
        ),

        "retrieval_p99": np.percentile(
            retrieval,
            99,
        ),

        "total_p50": np.percentile(
            total,
            50,
        ),

        "total_p90": np.percentile(
            total,
            90,
        ),

        "total_p95": np.percentile(
            total,
            95,
        ),

        "total_p99": np.percentile(
            total,
            99,
        ),
    }


st.set_page_config(
    page_title="Multilingual CPU RAG",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# DIRECTORIES
# ============================================================

PDF_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #777;
        font-size: 16px;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


stats = calculate_latency_percentiles()

if stats:

    st.subheader(
        "📈 Latency Percentiles"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "P50 Retrieval",
        f"{stats['retrieval_p50']:.0f} ms",
    )

    c2.metric(
        "P90 Retrieval",
        f"{stats['retrieval_p90']:.0f} ms",
    )

    c3.metric(
        "P95 Retrieval",
        f"{stats['retrieval_p95']:.0f} ms",
    )

    c4.metric(
        "P99 Retrieval",
        f"{stats['retrieval_p99']:.0f} ms",
    )

    st.caption(
        f"Based on {stats['count']} queries"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "P50 End-to-End",
        f"{stats['total_p50'] / 1000:.2f} s",
    )

    c2.metric(
        "P90 End-to-End",
        f"{stats['total_p90'] / 1000:.2f} s",
    )

    c3.metric(
        "P95 End-to-End",
        f"{stats['total_p95'] / 1000:.2f} s",
    )

    c4.metric(
        "P99 End-to-End",
        f"{stats['total_p99'] / 1000:.2f} s",
    )

# ============================================================
# RETRIEVER
# ============================================================

@st.cache_resource
def get_retriever():
    return HybridRetriever()


# ============================================================
# INGESTION FUNCTION
# ============================================================

def run_ingestion():

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    command = [
        sys.executable,
        "-m",
        "src.ingest",
    ]

    result = subprocess.run(
        command,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )

    return result


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '📚 Multilingual CPU RAG Chatbot'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'PyMuPDF + Tesseract + multilingual embeddings '
    '+ Qdrant HNSW + BM25/RRF'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 Corpus")

    # ========================================================
    # MULTIPLE PDF UPLOAD
    # ========================================================

    st.subheader("📤 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Choose multiple PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help=(
            "Select one or more PDFs. "
            "After selection, the files are automatically "
            "saved and ingestion starts."
        ),
    )

    # ========================================================
    # AUTOMATIC UPLOAD + INGESTION
    # ========================================================
# ========================================================
# AUTOMATIC INGESTION
# ========================================================

    if uploaded_files:

        upload_signature = tuple(
            sorted(
                (
                    file.name,
                    file.size,
                )
                for file in uploaded_files
            )
        )

        if (
            st.session_state.get("last_upload_signature")
            != upload_signature
        ):

            # Remember this upload
            st.session_state[
                "last_upload_signature"
            ] = upload_signature

            # ================================================
            # SAVE FILES
            # ================================================

            with st.spinner(
                "📥 Uploading PDF files..."
            ):

                saved_files = []

                for uploaded_file in uploaded_files:

                    filename = Path(
                        uploaded_file.name
                    ).name

                    output_path = (
                        PDF_DIR / filename
                    )

                    with open(
                        output_path,
                        "wb",
                    ) as f:

                        f.write(
                            uploaded_file.getbuffer()
                        )

                    saved_files.append(filename)

            st.success(
                f"✅ {len(saved_files)} PDF(s) uploaded."
            )

            # ================================================
            # INGESTION
            # ================================================

            with st.status(
                "🚀 Processing PDFs...",
                expanded=True,
            ) as ingestion_status:

                try:

                    st.write(
                        "📄 Reading PDF documents..."
                    )

                    st.write(
                        "🔎 Extracting native text..."
                    )

                    st.write(
                        "🖼️ Running OCR on scanned pages..."
                    )

                    st.write(
                        "🌐 Detecting document languages..."
                    )

                    st.write(
                        "🧹 Cleaning and normalizing text..."
                    )

                    st.write(
                        "✂️ Creating document chunks..."
                    )

                    st.write(
                        "🧠 Generating multilingual embeddings..."
                    )

                    st.write(
                        "🗄️ Updating Qdrant HNSW index..."
                    )

                    st.write(
                        "🔎 Updating BM25 index..."
                    )

                    result = run_ingestion()

                    if result.stdout:
                        st.code(
                            result.stdout,
                            language="text",
                        )

                    if result.stderr:
                        st.code(
                            result.stderr,
                            language="text",
                        )

                    if result.returncode == 0:

                        ingestion_status.update(
                            label="✅ PDF ingestion completed",
                            state="complete",
                        )

                        st.success(
                            "🎉 All uploaded PDFs "
                            "have been indexed successfully."
                        )

                        # New corpus → invalidate cached retriever
                        get_retriever.clear()

                        # Warm the new retriever immediately
                        get_retriever()

                    else:

                        ingestion_status.update(
                            label="❌ PDF ingestion failed",
                            state="error",
                        )

                        st.error(
                            f"Ingestion failed with "
                            f"exit code {result.returncode}."
                        )

                except Exception as exc:

                    ingestion_status.update(
                        label="❌ Ingestion crashed",
                        state="error",
                    )

                    st.error(
                        "An error occurred during ingestion."
                    )

                    st.exception(exc)
        # ========================================================
        # CURRENT PDF FILES
        # ========================================================

        st.divider()

        pdf_files = sorted(
            PDF_DIR.glob("*.pdf")
        )

        st.subheader(
            f"📄 PDF Files ({len(pdf_files)})"
        )

        if pdf_files:

            for pdf in pdf_files:

                size_mb = (
                    pdf.stat().st_size
                    / (1024 * 1024)
                )

                st.write(
                    f"📄 {pdf.name}"
                )

                st.caption(
                    f"{size_mb:.2f} MB"
                )

        else:

            st.info(
                "No PDFs uploaded yet."
            )

    # ========================================================
    # CORPUS STATISTICS
    # ========================================================

    st.divider()

    st.subheader(
        "📊 Corpus Statistics"
    )

    if REGISTRY_PATH.exists():

        try:

            registry = json.loads(
                REGISTRY_PATH.read_text(
                    encoding="utf-8"
                )
            )

            total_pdfs = len(
                registry
            )

            total_pages = sum(
                item.get(
                    "pages",
                    0
                )
                for item in registry.values()
            )

            total_chunks = sum(
                item.get(
                    "chunks",
                    0
                )
                for item in registry.values()
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "PDFs",
                total_pdfs
            )

            c2.metric(
                "Pages",
                total_pages
            )

            c3.metric(
                "Chunks",
                total_chunks
            )

        except Exception as exc:

            st.warning(
                f"Could not read registry: {exc}"
            )

    else:

        st.info(
            "No ingestion registry yet."
        )

    # ========================================================
    # SYSTEM INFORMATION
    # ========================================================

    st.divider()

    st.subheader(
        "🔧 System"
    )

    st.caption(
        f"Qdrant: {QDRANT_HOST}:{QDRANT_PORT}"
    )

    st.caption(
        f"Collection: {QDRANT_COLLECTION}"
    )

    st.caption(
        "Embedding: "
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    st.caption(
        "Retrieval: Qdrant HNSW + BM25 + RRF"
    )

    st.caption(
        f"LLM: {GROQ_MODEL}"
    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about the PDFs..."
)


# ============================================================
# QUESTION PROCESSING
# ============================================================

if question:

    # ========================================================
    # TOTAL REQUEST TIMER
    # ========================================================

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # USER QUESTION
    # --------------------------------------------------------

    with st.chat_message("user"):
        st.write(question)

    # --------------------------------------------------------
    # RETRIEVAL + GENERATION
    # --------------------------------------------------------

    try:

        with st.chat_message("assistant"):

            # ================================================
            # RETRIEVER INITIALIZATION
            # ================================================

            init_start = time.perf_counter()

            retriever = get_retriever()

            retriever_init_ms = (
                time.perf_counter() - init_start
            ) * 1000

            # ================================================
            # ACTUAL RETRIEVAL
            # ================================================

            with st.spinner(
                "🔎 Retrieving relevant documents..."
            ):

                search_start = time.perf_counter()

                results = retriever.search(
                    question,
                    top_k=5,
                )

                search_ms = (
                    time.perf_counter() - search_start
                ) * 1000

            # ================================================
            # GENERATION
            # ================================================

            generation_start = time.perf_counter()

            if not results:

                answer = (
                    "I could not find relevant "
                    "information in the indexed documents."
                )

            else:

                with st.spinner(
                    "🤖 Generating answer..."
                ):

                    answer = generate_answer(
                        question,
                        results,
                    )

            generation_ms = (
                time.perf_counter()
                - generation_start
            ) * 1000

            # ================================================
            # TOTAL
            # ================================================

            total_ms = (
                time.perf_counter()
                - total_start
            ) * 1000
            save_query_metrics(
                retrieval_ms=search_ms,
                generation_ms=generation_ms,
                total_ms=total_ms,
            )


            # ================================================
            # ANSWER
            # ================================================

            st.markdown(answer)

        # ====================================================
        # PERFORMANCE
        # ====================================================

        st.divider()

        st.subheader(
            "⏱️ Query Performance"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Retriever Init",
                f"{retriever_init_ms:.0f} ms"
            )

        with c2:

            st.metric(
                "Retrieval",
                f"{search_ms:.0f} ms"
            )

        with c3:

            st.metric(
                "Generation",
                f"{generation_ms / 1000:.2f} s"
            )

        with c4:

            st.metric(
                "End-to-End",
                f"{total_ms / 1000:.2f} s"
            )

        # ====================================================
        # SOURCES
        # ====================================================

        st.subheader(
            "📚 Sources / Retrieved Chunks"
        )

        if not results:

            st.info(
                "No sources retrieved."
            )

        else:

            for i, result in enumerate(
                results,
                start=1,
            ):

                payload = result.get(
                    "payload",
                    {},
                )

                filename = payload.get(
                    "filename",
                    "Unknown",
                )

                page = payload.get(
                    "page",
                    "?",
                )

                score = result.get(
                    "score",
                    0,
                )

                language = payload.get(
                    "language",
                    "unknown",
                )

                source_type = payload.get(
                    "source_type",
                    "native",
                )

                text = payload.get(
                    "text",
                    "",
                )

                with st.expander(
                    f"{i}. {filename} "
                    f"— Page {page} "
                    f"— RRF {score:.5f}"
                ):

                    st.caption(
                        f"Language: {language} | "
                        f"Source: {source_type}"
                    )

                    st.write(text)

    except Exception as exc:

        st.error(
            "❌ Error while processing the question."
        )

        with st.expander(
            "Technical details"
        ):

            st.exception(
                exc
            )
