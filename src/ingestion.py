import json
import os
import urllib.request
import arxiv
from pypdf import PdfReader
from tqdm import tqdm

TARGET_PAPERS = [
    {"arxiv_id": "1706.03762", "topic": "Transformer Architecture", "label": "Attention Is All You Need"},
    {"arxiv_id": "1512.03385", "topic": "Computer Vision", "label": "Deep Residual Learning (ResNet)"},
    {"arxiv_id": "1810.04805", "topic": "NLP / Pre-training", "label": "BERT"},
    {"arxiv_id": "2005.14165", "topic": "Large Language Models", "label": "Language Models are Few-Shot Learners (GPT-3)"},
    {"arxiv_id": "1406.2661",  "topic": "Generative Models", "label": "Generative Adversarial Nets (GAN)"},
    {"arxiv_id": "2006.11239", "topic": "Generative Models", "label": "Denoising Diffusion Models (DDPM)"},
    {"arxiv_id": "2106.09685", "topic": "Model Fine-tuning", "label": "LoRA: Low-Rank Adaptation"},
    {"arxiv_id": "2005.11401", "topic": "RAG Foundations", "label": "Retrieval-Augmented Generation (RAG)"},
    {"arxiv_id": "1412.6980",  "topic": "Optimization", "label": "Adam Optimizer"},
    {"arxiv_id": "1404.5997",  "topic": "Convolutional Networks", "label": "AlexNet"}
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PDF_DIR = os.path.join(BASE_DIR, "data", "raw_pdfs")
PROCESSED_JSON = os.path.join(BASE_DIR, "data", "processed_chunks.json")


def download_and_parse_papers():
    os.makedirs(RAW_PDF_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PROCESSED_JSON), exist_ok=True)
    
    client = arxiv.Client()
    parsed_documents = []

    print("Fetching metadata and PDFs from arXiv...")
    for item in tqdm(TARGET_PAPERS, desc="Processing Papers"):
        arxiv_id = item["arxiv_id"]
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(client.results(search))

        pdf_filename = f"{arxiv_id.replace('/', '_')}.pdf"
        pdf_path = os.path.join(RAW_PDF_DIR, pdf_filename)

        # Download using standard urlretrieve if not already present
        if not os.path.exists(pdf_path):
            pdf_url = paper.pdf_url
            if not pdf_url.endswith(".pdf"):
                pdf_url += ".pdf"
            urllib.request.urlretrieve(pdf_url, pdf_path)

        # Extract text per page
        reader = PdfReader(pdf_path)
        extracted_pages = []
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                extracted_pages.append({
                    "page_number": idx + 1,
                    "text": text.strip()
                })

        parsed_documents.append({
            "arxiv_id": arxiv_id,
            "title": paper.title,
            "topic": item["topic"],
            "published_year": paper.published.year,
            "authors": [a.name for a in paper.authors],
            "abstract": paper.summary,
            "pages": extracted_pages
        })

    with open(PROCESSED_JSON, "w", encoding="utf-8") as f:
        json.dump(parsed_documents, f, indent=2, ensure_ascii=False)

    print(f"\nProcessing complete: {len(parsed_documents)} papers saved to {PROCESSED_JSON}")


if __name__ == "__main__":
    download_and_parse_papers()
