import os
import logging
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.readers.file import UnstructuredReader
import chromadb

# Import custom LLM utilities
from llm_utils import GigaChatLLM, setup_terminal_encoding

# Setup logging and encoding
setup_terminal_encoding()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def build_advanced_rag_pipeline(pdf_path: str, db_path: str = "./chroma_db", collection_name: str = "advanced_rag"):
    # 0. Set GigaChat LLM for generation
    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    scope = os.getenv("GIGACHAT_SCOPE")
    
    if not credentials:
        raise ValueError("GIGACHAT_CREDENTIALS must be set in .env")
        
    Settings.llm = GigaChatLLM(credentials=credentials, scope=scope)
    
    # 1. Initialize Local Models (Embeddings & Reranking)
    print("Loading local embedding model (BGE-M3)...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    
    print("Loading local reranker (BGE-Reranker-Base)...")
    rerank_postprocessor = SentenceTransformerRerank(
        model="BAAI/bge-reranker-base", 
        top_n=3
    )

    # 2. Configuration for Sentence Window
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    # 3. Load PDF
    print(f"Loading PDF from {pdf_path}...")
    documents = UnstructuredReader().load_data(file=pdf_path)

    # 4. Initialize ChromaDB
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 5. Build Index
    nodes = node_parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    # 6. Configure Query Engine
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window"),
            rerank_postprocessor
        ],
    )

    return query_engine

if __name__ == "__main__":
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"File '{pdf_file}' not found. Usage: python rag_pipeline.py [path_to_pdf] [query]")
        sys.exit(1)
        
    try:
        engine = build_advanced_rag_pipeline(pdf_file)
        query = sys.argv[2] if len(sys.argv) > 2 else "What is the main topic of this document?"
        
        response = engine.query(query)
        print(f"\nQUERY: {query}\nRESPONSE: {response}\n")
    except Exception as e:
        print(f"An error occurred: {e}")
