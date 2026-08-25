import os
import sys
import time
from typing import Dict, Any, List, TypedDict

# Ensure root directory is in sys.path regardless of execution context
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from src.langchain_engine import SharedVectorRetriever, format_docs

load_dotenv()


# 1. State Definition
class GraphState(TypedDict):
    question: str
    transformed_query: str
    documents: List[Document]
    citations: List[Dict[str, Any]]
    generation: str
    retry_count: int


# 2. Document Relevance Evaluator Model
class GradeRelevance(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")


class LangGraphCorrectiveRAG:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set.")

        self.retriever = SharedVectorRetriever()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=api_key
        )
        self.app = self._build_graph()

    def _get_single_query_embedding(self, text: str) -> List[float]:
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
        embed_model = GoogleGenAIEmbedding(
            model_name="gemini-embedding-001",
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        return embed_model.get_query_embedding(text)

    # --- Node: Retrieve ---
    def retrieve_node(self, state: GraphState) -> Dict[str, Any]:
        query = state.get("transformed_query") or state["question"]
        query_vec = self._get_single_query_embedding(query)
        results = self.retriever.similarity_search_with_score(query_vec, k=4)

        docs = [doc for doc, _ in results]
        citations = []
        for doc, score in results:
            citations.append({
                "title": doc.metadata.get("title", "Unknown"),
                "topic": doc.metadata.get("topic", "General"),
                "page": doc.metadata.get("page_number", "N/A"),
                "score": round(score, 4),
                "snippet": doc.page_content[:200] + "..."
            })
        return {"documents": docs, "citations": citations}

    # --- Node: Grade Documents ---
    def grade_documents_node(self, state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        docs = state["documents"]

        grader_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a grader assessing relevance of a retrieved document to a user question. "
                       "Respond ONLY with 'yes' if the document contains keywords or semantic info related to the question, or 'no' otherwise."),
            ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}")
        ])

        grader_chain = grader_prompt | self.llm.with_structured_output(GradeRelevance)
        filtered_docs = []
        for d in docs:
            res = grader_chain.invoke({"question": question, "document": d.page_content})
            if res.binary_score.strip().lower() == "yes":
                filtered_docs.append(d)

        # Fallback to keep top document if all were graded as 'no'
        if not filtered_docs and docs:
            filtered_docs = [docs[0]]

        return {"documents": filtered_docs}

    # --- Node: Rewrite Query ---
    def rewrite_query_node(self, state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        retry_count = state.get("retry_count", 0) + 1

        re_write_prompt = ChatPromptTemplate.from_template(
            "Look at the input question and optimize it to be an effective technical search query for Deep Learning literature:\n\n"
            "Initial Question: {question}\n\nImproved Search Query:"
        )
        rewrite_chain = re_write_prompt | self.llm | StrOutputParser()
        better_query = rewrite_chain.invoke({"question": question})
        return {"transformed_query": better_query.strip(), "retry_count": retry_count}

    # --- Node: Generate ---
    def generate_node(self, state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        docs = state["documents"]

        prompt = ChatPromptTemplate.from_template(
            "You are an expert AI research assistant specializing in Deep Learning papers.\n"
            "Use ONLY the following context to answer the question:\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Comprehensive Technical Answer:"
        )
        rag_chain = prompt | self.llm | StrOutputParser()
        generation = rag_chain.invoke({"context": format_docs(docs), "question": question})
        return {"generation": generation}

    # --- Conditional Edge ---
    def decide_to_generate(self, state: GraphState) -> str:
        docs = state.get("documents", [])
        retry_count = state.get("retry_count", 0)

        # If we have at least 2 relevant docs or already retried once, generate
        if len(docs) >= 2 or retry_count >= 1:
            return "generate"
        return "rewrite_query"

    def _build_graph(self):
        workflow = StateGraph(GraphState)

        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("grade_documents", self.grade_documents_node)
        workflow.add_node("rewrite_query", self.rewrite_query_node)
        workflow.add_node("generate", self.generate_node)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "generate": "generate",
                "rewrite_query": "rewrite_query"
            }
        )
        workflow.add_edge("rewrite_query", "retrieve")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def query(self, prompt: str) -> Dict[str, Any]:
        start_time = time.time()
        initial_state: GraphState = {
            "question": prompt,
            "transformed_query": "",
            "documents": [],
            "citations": [],
            "generation": "",
            "retry_count": 0
        }

        final_state = self.app.invoke(initial_state)
        latency = round(time.time() - start_time, 2)

        return {
            "engine": "LangGraph (Corrective RAG)",
            "answer": final_state["generation"],
            "latency_seconds": latency,
            "transformed_query": final_state.get("transformed_query", None),
            "citations": final_state["citations"]
        }


if __name__ == "__main__":
    print("Testing LangGraph Corrective RAG Engine...")
    crag = LangGraphCorrectiveRAG()
    sample_query = "What is the primary motivation for introducing residual connections in ResNet?"
    result = crag.query(sample_query)

    print("\n--- Corrective RAG Answer ---")
    print(result["answer"])
    print(f"\nLatency: {result['latency_seconds']}s")
    if result["transformed_query"]:
        print(f"Transformed Query: {result['transformed_query']}")
    print("\n--- Citations ---")
    for c in result["citations"]:
        print(f"- [{c['title']}, Page {c['page']}] (Score: {c['score']})")
