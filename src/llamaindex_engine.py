import json
import os
import time
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_JSON = os.path.join(BASE_DIR, "data", "processed_chunks.json")
STORAGE_DIR = os.path.join(BASE_DIR, "chroma_db", "llamaindex_storage")


class RateLimitedGoogleEmbedding(GoogleGenAIEmbedding):
    """Custom wrapper to respect Google GenAI 15 RPM free tier quota."""
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        batch_size = 5
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            success = False
            retries = 0
            while not success and retries < 5:
                try:
                    batch_embeds = super()._get_text_embeddings(batch)
                    embeddings.extend(batch_embeds)
                    success = True
                    time.sleep(2.0)  # Gentle delay between requests to stay well below 15 RPM
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        retries += 1
                        wait_time = 15 * retries
                        print(f"\n[Rate Limit] Exceeded quota. Sleeping for {wait_time}s before retry {retries}/5...")
                        time.sleep(wait_time)
                    else:
                        raise e
        return embeddings


def init_llamaindex_settings():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    Settings.llm = GoogleGenAI(
        model="gemini-3.6-flash",
        api_key=api_key,
        temperature=0.1
    )
    Settings.embed_model = RateLimitedGoogleEmbedding(
        model_name="gemini-embedding-001",
        api_key=api_key
    )
    # Larger chunks reduce total API calls by ~50%
    Settings.node_parser = SentenceSplitter(chunk_size=2048, chunk_overlap=128)


def load_documents_from_json() -> List[Document]:
    with open(PROCESSED_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for paper in data:
        for page in paper["pages"]:
            doc = Document(
                text=page["text"],
                metadata={
                    "title": paper["title"],
                    "arxiv_id": paper["arxiv_id"],
                    "topic": paper["topic"],
                    "published_year": paper["published_year"],
                    "page_number": page["page_number"]
                },
                excluded_embed_metadata_keys=["page_number", "arxiv_id"],
                excluded_llm_metadata_keys=["arxiv_id"]
            )
            documents.append(doc)
    return documents


def get_or_create_llamaindex():
    init_llamaindex_settings()
    
    if os.path.exists(STORAGE_DIR) and os.listdir(STORAGE_DIR):
        print("Loading existing LlamaIndex from disk...")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        return load_index_from_storage(storage_context)

    print("Building new LlamaIndex with rate-limiting and saving to disk...")
    documents = load_documents_from_json()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    os.makedirs(STORAGE_DIR, exist_ok=True)
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    return index


class LlamaIndexAssistant:
    def __init__(self):
        self.index = get_or_create_llamaindex()

    def query(self, prompt: str, topic_filter: Optional[str] = None, similarity_top_k: int = 4) -> Dict[str, Any]:
        start_time = time.time()

        filters = None
        if topic_filter:
            filters = MetadataFilters(
                filters=[MetadataFilter(key="topic", value=topic_filter, operator=FilterOperator.EQ)]
            )

        query_engine = self.index.as_query_engine(
            similarity_top_k=similarity_top_k,
            filters=filters
        )

        response = query_engine.query(prompt)
        latency = round(time.time() - start_time, 2)

        citations = []
        for node in response.source_nodes:
            citations.append({
                "title": node.metadata.get("title", "Unknown"),
                "topic": node.metadata.get("topic", "General"),
                "page": node.metadata.get("page_number", "N/A"),
                "score": round(node.score, 4) if node.score is not None else None,
                "snippet": node.text[:200] + "..."
            })

        return {
            "engine": "LlamaIndex",
            "answer": response.response,
            "latency_seconds": latency,
            "citations": citations
        }


if __name__ == "__main__":
    print("Testing LlamaIndex Assistant...")
    assistant = LlamaIndexAssistant()
    sample_query = "What is the primary motivation for introducing residual connections in ResNet?"
    result = assistant.query(sample_query)
    
    print("\n--- Answer ---")
    print(result["answer"])
    print(f"\nLatency: {result['latency_seconds']}s")
    print("\n--- Citations ---")
    for c in result["citations"]:
        print(f"- [{c['title']}, Page {c['page']}] (Score: {c['score']})")
