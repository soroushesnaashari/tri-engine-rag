import os
import sys
import time
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.llamaindex_engine import LlamaIndexAssistant
from src.langchain_engine import LangChainAssistant
from src.langgraph_engine import LangGraphCorrectiveRAG

st.set_page_config(
    page_title="Tri-Engine Deep Learning RAG Assistant",
    page_icon="🧠",
    layout="wide"
)

# Ultra-Minimal Monochrome CSS
st.markdown("""
<style>
    /* Global monochrome typography */
    body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #111111;
    }
    
    /* Subtle container borders and spacing */
    .stTextArea textarea {
        font-size: 14px;
        border-radius: 4px;
        border: 1px solid #cccccc;
    }
    .stTextArea textarea:focus {
        border-color: #000000;
        box-shadow: none;
    }
    
    /* Compact clean button */
    .stButton button {
        background-color: #000000;
        color: #ffffff;
        border: 1px solid #000000;
        border-radius: 4px;
        padding: 6px 18px;
        font-weight: 500;
        font-size: 14px;
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        background-color: #ffffff;
        color: #000000;
        border: 1px solid #000000;
    }
    
    /* Citation block */
    .citation-card {
        border-left: 2px solid #000000;
        padding-left: 12px;
        margin-top: 12px;
        margin-bottom: 12px;
        font-size: 13px;
        color: #333333;
    }
    
    /* Divider cleanup */
    hr {
        border: 0;
        border-top: 1px solid #e0e0e0;
        margin: 28px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engines():
    llama = LlamaIndexAssistant()
    langchain = LangChainAssistant()
    langgraph = LangGraphCorrectiveRAG()
    return llama, langchain, langgraph

with st.spinner("Initializing system engines..."):
    llama_engine, langchain_engine, langgraph_engine = load_engines()

# Header & Developer Attribution
st.title("Tri-Engine Deep Learning RAG Assistant")
st.markdown(
    "A knowledge assistant benchmarking **LlamaIndex**, **LangChain** and **LangGraph** over 10 seminal Deep Learning papers developed by [Mohammad Soroush Esnaashari](https://soroushesnaashari.github.io/)."
)

st.write("")

# Sidebar Configuration
st.sidebar.markdown("## Settings")
mode = st.sidebar.selectbox(
    "Execution Mode",
    ["Single Engine", "Compare All Engines Side-by-Side"]
)

selected_engine = None
if mode == "Single Engine":
    selected_engine = st.sidebar.radio(
        "Engine Selection",
        ["LangGraph (Corrective RAG)", "LangChain (LCEL)", "LlamaIndex"]
    )

topic_filter = st.sidebar.selectbox(
    "Domain Filter (LlamaIndex)",
    [
        "All Topics",
        "Transformer Architecture",
        "Computer Vision",
        "NLP / Pre-training",
        "Large Language Models",
        "Generative Models",
        "Model Fine-tuning",
        "RAG Foundations",
        "Optimization",
        "Convolutional Networks"
    ]
)
actual_topic = None if topic_filter == "All Topics" else topic_filter

sample_queries = [
    "What is the primary motivation for introducing residual connections in ResNet?",
    "Why does the Transformer use Scaled Dot-Product attention instead of Additive attention?",
    "What two pre-training objectives are used in BERT, and why?",
    "How does LoRA reduce trainable parameters during fine-tuning?",
    "What is the mathematical formulation of the Adam optimizer?"
]

st.sidebar.markdown("---")
st.sidebar.markdown("### Benchmark Questions")
selected_sample = st.sidebar.selectbox("Load Sample Query", ["-- Select a query --"] + sample_queries)

# Main Query Layout with generous whitespace
col_input, col_space = st.columns([3, 1])

with col_input:
    user_query = st.text_area(
        "Research Question",
        value="" if selected_sample == "-- Select a query --" else selected_sample,
        height=75,
        placeholder="Enter research inquiry..."
    )
    
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        run_pressed = st.button("Execute")

def display_result(col, result):
    with col:
        st.markdown(f"#### {result['engine']}")
        st.caption(f"Latency: {result['latency_seconds']}s")
        
        if result.get("transformed_query"):
            st.caption(f"Transformed Query: {result['transformed_query']}")

        st.markdown(result["answer"])

        with st.expander("Retrieved Source Citations"):
            for idx, c in enumerate(result.get("citations", []), 1):
                st.markdown(f"""
                <div class="citation-card">
                    <strong>{idx}. {c['title']}</strong> (Page {c['page']}) &bull; <em>{c['topic']}</em> &bull; Score: {c['score']}<br>
                    {c['snippet']}
                </div>
                """, unsafe_allow_html=True)

# Output Execution
if run_pressed:
    if not user_query.strip():
        st.warning("Please provide a question.")
    else:
        st.markdown("---")
        if mode == "Single Engine":
            with st.spinner("Processing query..."):
                if selected_engine == "LlamaIndex":
                    res = llama_engine.query(user_query, topic_filter=actual_topic)
                elif selected_engine == "LangChain (LCEL)":
                    res = langchain_engine.query(user_query)
                else:
                    res = langgraph_engine.query(user_query)

            container = st.container()
            display_result(container, res)

        else:
            st.markdown("### Performance Comparison")
            col1, col2, col3 = st.columns(3)

            with st.spinner("Executing all 3 engines..."):
                res_llama = llama_engine.query(user_query, topic_filter=actual_topic)
                res_langchain = langchain_engine.query(user_query)
                res_langgraph = langgraph_engine.query(user_query)

            display_result(col1, res_llama)
            display_result(col2, res_langchain)
            display_result(col3, res_langgraph)

            st.markdown("---")
            st.markdown("### Summary Statistics")
            st.table([
                {"Engine": "LlamaIndex", "Latency (s)": res_llama["latency_seconds"], "Citations": len(res_llama["citations"])},
                {"Engine": "LangChain (LCEL)", "Latency (s)": res_langchain["latency_seconds"], "Citations": len(res_langchain["citations"])},
                {"Engine": "LangGraph (CRAG)", "Latency (s)": res_langgraph["latency_seconds"], "Citations": len(res_langgraph["citations"])}
            ])
