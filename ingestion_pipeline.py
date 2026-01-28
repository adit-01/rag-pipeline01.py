import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import shutil
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
import torch


def load_documents(docs_path="DOCS"):
    print(f"Loading documents from: {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")

    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf8"}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}")

    print(f"Loaded {len(documents)} documents")
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks


class CustomEmbeddings:
    """Custom embedding function using SentenceTransformer directly"""
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully")
    
    def embed_documents(self, texts):
        """Embed a list of documents"""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def embed_query(self, text):
        """Embed a single query"""
        embedding = self.model.encode([text])[0]
        return embedding.tolist()


def create_vector_store(chunks, persist_directory="chroma_db"):
    if os.path.exists(persist_directory):
        print(f"Deleting existing vector store at {persist_directory}...")
        shutil.rmtree(persist_directory)
        print("Old vector store deleted")
    
    print("Creating embeddings and storing in ChromaDB...")

    # Use custom embeddings to avoid any OpenAI fallback
    embeddings = CustomEmbeddings("sentence-transformers/all-MiniLM-L6-v2")

    print("Creating vector store...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Vector store created and saved to: {persist_directory}")
    return vectorstore


def main():
    print("Starting vector store creation...")
    
    try:
        documents = load_documents(docs_path="DOCS")
        chunks = split_documents(documents)
        vector_store = create_vector_store(chunks, persist_directory="chroma_db")
        
        print("\nVector store creation complete!")
        print(f"Total documents processed: {len(documents)}")
        print(f"Total chunks created: {len(chunks)}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()