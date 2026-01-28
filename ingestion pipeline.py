import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import    CharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv, dotenv_values

load_dotenv()


def load_documents(docs_path="DOCS"):
    """load all text files from the docs directory"""
    print(f"loading documents from: {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. please create it and add your company files.")
    


    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf8"},
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in the specified path: {docs_path}. please add your company documents.")

    for i, doc in enumerate(documents[:2]):
        print("\nDocuments[:2]:")
        print(f" source: {doc.metadata['source']}")
        print(f" content length: {len(doc.page_content)} characters")
        print(f" content preview: {doc.page_content[:100]}...")
        print(f"metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=800, chunk_overlap=0):   

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"source: {chunk.metadata['source']}")
            print(f"length: {len(chunk.page_content)} characters")
            print(f"content:")
            print(chunk.page_content)
            print("-" * 50)

        if len(chunks) > 5:
            print(f"\n... {len(chunks) - 5} more chunks")


    return chunks

def create_vector_store(chunks, persist_directory="chroma_db"):
    """create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")

    # Import here so the OpenAI client reads the API key we force-set above
    from langchain_openai import OpenAIEmbeddings

    # Debug: confirm env var presence (do not print the key itself)
    _key_present = bool(os.getenv("OPENAI_API_KEY"))
    print(f"OPENAI_API_KEY present in environment: {_key_present}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    print(f"vector store created and saved to: {persist_directory}")
    return vectorstore

def main():
    print("Main Function")

    documents = load_documents(docs_path="DOCS")

    env_path = os.path.join(os.path.dirname(__file__), ".env")
    # Also parse the file directly and force the value into os.environ to guarantee availability
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k == "OPENAI_API_KEY" or k == "OPEN_AI_KEY":
                        os.environ["OPENAI_API_KEY"] = v
                        break
        except Exception:
            pass
    else:
        # fallback to dotenv_values if file not found in script dir
        try:
            cfg = dotenv_values()
            key = cfg.get("OPENAI_API_KEY") or cfg.get("OPEN_AI_KEY")
            if key:
                os.environ["OPENAI_API_KEY"] = key
        except Exception:
            pass

    chunks = split_documents(documents)

    vector_store = create_vector_store(chunks,)


if __name__ == "__main__":
    main()
