import json
import os
import time
import numpy as np
from typing import Dict, Any, List, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_DIR = os.path.join(BASE_DIR, "chroma_db", "llamaindex_storage")
DEFAULT_VECTOR_STORE = os.path.join(STORAGE_DIR, "default__vector_store.json")
DOCSTORE = os.path.join(STORAGE_DIR, "docstore.json")


class SharedVectorRetriever:
    """Fast in-memory semantic retriever that reuses precomputed embeddings from storage."""

    def __init__(self):
        self.node_ids = []
        self.vectors = []
        self.documents_map = {}
        self._load_persisted_storage()

    def _load_persisted_storage(self):
        if not os.path.exists(DEFAULT_VECTOR_STORE) or not os.path.exists(DOCSTORE):
            raise FileNotFoundError(
                "Persisted embeddings not found in chroma_db/llamaindex_storage. "
                "Please ensure Phase 2 (llamaindex_engine.py) has run first."
            )

        with open(DEFAULT_VECTOR_STORE, "r", encoding="utf-8") as f:
            vstore_data = json.load(f)

        with open(DOCSTORE, "r", encoding="utf-8") as f:
            docstore_data = json.load(f)

        embedding_dict = vstore_data.get("embedding_dict", {})
        doc_dict = docstore_data.get("docstore/data", {})

        for node_id, vector in embedding_dict.items():
            if node_id in doc_dict:
                node_data = doc_dict[node_id]["__data__"]
                text = node_data.get("text", "")
                metadata = node_data.get("metadata", {})
                
                self.node_ids.append(node_id)
                self.vectors.append(vector)
                self.documents_map[node_id] = Document(
                    page_content=text,
                    metadata=metadata
                )

        self.vectors = np.array(self.vectors, dtype=np.float32)
        # Normalize vectors for fast cosine similarity via dot product
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.vectors = self.vectors / norms

    def similarity_search_with_score(self, query_embedding: List[float], k: int = 4) -> List[Tuple[Document, float]]:
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scores = np.dot(self.vectors, q_vec)
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            node_id = self.node_ids[idx]
            results.append((self.documents_map[node_id], float(scores[idx])))
        return results


def format_docs(docs: List[Document]) -> str:
    formatted = []
    for doc in docs:
        header = f"--- Document: {doc.metadata.get('title', 'Unknown')} (Page {doc.metadata.get('page_number', 'N/A')}) ---"
        formatted.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(formatted)


class LangChainAssistant:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set.")

        self.retriever = SharedVectorRetriever()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=api_key,
            temperature=0.1
        )
        self.prompt_template = ChatPromptTemplate.from_template(
            """You are an expert AI research assistant specializing in Deep Learning papers.
Use ONLY the provided context to answer the question. If the context does not contain enough information, state that clearly.

Context:
{context}

Question: {question}

Provide a comprehensive, technical answer with clear explanations:"""
        )

    def _get_single_query_embedding(self, text: str) -> List[float]:
        """Embeds only the single user prompt (1 API call)."""
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
        embed_model = GoogleGenAIEmbedding(
            model_name="gemini-embedding-001",
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        return embed_model.get_query_embedding(text)

    def query(self, prompt: str, k: int = 4) -> Dict[str, Any]:
        start_time = time.time()

        # Step 1: Embed single query and retrieve from pre-indexed memory
        query_vec = self._get_single_query_embedding(prompt)
        retrieved_docs_and_scores = self.retriever.similarity_search_with_score(query_vec, k=k)
        retrieved_docs = [doc for doc, _ in retrieved_docs_and_scores]

        # Step 2: LCEL Runnable Chain
        rag_chain = (
            {"context": lambda _: format_docs(retrieved_docs), "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

        answer = rag_chain.invoke(prompt)
        latency = round(time.time() - start_time, 2)

        citations = []
        for doc, score in retrieved_docs_and_scores:
            citations.append({
                "title": doc.metadata.get("title", "Unknown"),
                "topic": doc.metadata.get("topic", "General"),
                "page": doc.metadata.get("page_number", "N/A"),
                "score": round(score, 4),
                "snippet": doc.page_content[:200] + "..."
            })

        return {
            "engine": "LangChain LCEL",
            "answer": answer,
            "latency_seconds": latency,
            "citations": citations
        }


if __name__ == "__main__":
    print("Testing LangChain LCEL Assistant (Reusing Persisted Embeddings)...")
    assistant = LangChainAssistant()
    sample_query = "What is the primary motivation for introducing residual connections in ResNet?"
    result = assistant.query(sample_query)

    print("\n--- Answer ---")
    print(result["answer"])
    print(f"\nLatency: {result['latency_seconds']}s")
    print("\n--- Citations ---")
    for c in result["citations"]:
        print(f"- [{c['title']}, Page {c['page']}] (Score: {c['score']})")
