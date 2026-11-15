"""
Simple Streamlit UI for reRankQ demo.
Provides a web interface for querying the RAG system.
"""

import streamlit as st
import requests
import json
from pathlib import Path

# Page config
st.set_page_config(
    page_title="reRankQ - Hybrid Search & RAG",
    page_icon="🔍",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .citation-box {
        background-color: #f0f2f6;
        border-left: 4px solid #4a5568;
        padding: 10px;
        margin: 5px 0;
        border-radius: 4px;
    }
    .metric-box {
        background-color: #e8f4f8;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def query_rag(query: str, top_k: int = 5, use_reranking: bool = True):
    """Query the RAG system."""
    try:
        response = requests.post(
            "http://localhost:8000/query",
            json={
                "query": query,
                "top_k": top_k,
                "use_reranking": use_reranking,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error querying RAG system: {e}")
        return None


def check_health():
    """Check system health."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.json()
    except:
        return None


def render_result(result: dict) -> None:
    """Render a query result (metrics, answer, citations)."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Latency", f"{result['latency_ms']:.0f}ms")
    with col2:
        st.metric("Chunks Used", result['chunks_used'])
    with col3:
        st.metric("Method", result['method'])

    st.divider()

    # Answer
    st.subheader("Answer")
    st.markdown(result['answer'])

    # Citations
    if result.get('citations'):
        st.subheader("Sources")
        for i, citation in enumerate(result['citations'], 1):
            with st.expander(f"[{i}] {citation['title']}"):
                st.markdown(f"**Section:** {citation['section']}")
                st.markdown(f"**URL:** [{citation['url']}]({citation['url']})")


def main():
    """Main Streamlit app."""
    st.title("🔍 reRankQ")
    st.caption("Hybrid Search & RAG with Evaluation")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)
        use_reranking = st.checkbox("Use reranking", value=True)
        
        st.divider()
        st.header("System Status")
        
        health = check_health()
        if health:
            st.success(f"Status: {health['status']}")
            st.json(health['components'])
        else:
            st.warning("Could not connect to backend")
        
        st.divider()
        st.header("About")
        st.markdown("""
        reRankQ is a portfolio-grade hybrid search and RAG system.
        
        **Features:**
        - BM25 + Dense retrieval
        - Reciprocal Rank Fusion
        - Cross-encoder reranking
        - Full evaluation harness
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Query", "Results", "Evaluation"])
    
    with tab1:
        st.header("Query the RAG System")
        
        query = st.text_area(
            "Enter your question about Stripe's API:",
            placeholder="How do I create a payment intent?",
            height=100,
        )
        
        if st.button("Search", type="primary"):
            if query:
                with st.spinner("Searching..."):
                    result = query_rag(query, top_k, use_reranking)
                
                if result:
                    st.session_state.current_result = result
                    # Show the answer right here: the Results tab is a separate
                    # view, so leaving it only there looks like no response.
                    st.divider()
                    render_result(result)
            else:
                st.warning("Please enter a query")
    
    with tab2:
        st.header("Search Results")
        
        if "current_result" in st.session_state:
            render_result(st.session_state.current_result)
        else:
            st.info("Enter a query to see results")
    
    with tab3:
        st.header("Evaluation Results")
        
        # Check for evaluation results
        results_dir = Path("results")
        
        if (results_dir / "ablation_results.json").exists():
            with open(results_dir / "ablation_results.json") as f:
                ablation = json.load(f)
            
            st.subheader("Ablation Study Results")
            
            # Create table
            if "methods" in ablation:
                methods = ablation["methods"]
                data = []
                for method_name, method_data in methods.items():
                    if method_data.get("status") != "skipped":
                        data.append({
                            "Method": method_name,
                            "MRR": f"{method_data['mrr']:.4f}",
                            "Recall@5": f"{method_data['recall_at_k'].get(5, 0):.4f}",
                            "nDCG@5": f"{method_data['ndcg_at_k'].get(5, 0):.4f}",
                        })
                
                st.dataframe(data, width="stretch")
        else:
            st.info("Run the ablation study to see evaluation results")
        
        # E2E evaluation
        if (results_dir / "e2e_eval_aggregate.json").exists():
            with open(results_dir / "e2e_eval_aggregate.json") as f:
                e2e = json.load(f)
            
            st.subheader("End-to-End Evaluation")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Faithfulness", f"{e2e['faithfulness']['mean']:.2f}")
            with col2:
                st.metric("Relevance", f"{e2e['relevance']['mean']:.2f}")
            with col3:
                st.metric("Overall", f"{e2e['overall']['mean']:.2f}")


if __name__ == "__main__":
    main()
