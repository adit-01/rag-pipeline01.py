import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import SentenceTransformer


class CustomEmbeddings:
    """Custom embedding function using SentenceTransformer directly"""
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    
    def embed_query(self, text):
        embedding = self.model.encode([text])[0]
        return embedding.tolist()


def load_vector_store(persist_directory="chroma_db"):
    """Load the existing ChromaDB vector store"""
    
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"Vector store not found at {persist_directory}. "
            "Please run the ingestion pipeline first."
        )
    
    print(f"Loading vector store from: {persist_directory}")
    
    embeddings = CustomEmbeddings("sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    print("✅ Vector store loaded successfully")
    return vectorstore


def format_docs(docs):
    """Format documents for context"""
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(vectorstore, k=4):
    """Create RAG chain using Ollama (FREE local LLM)"""
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    # Create prompt template
    template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Initialize Ollama LLM (FREE - runs locally)
    print("Initializing Ollama LLM...")
    llm = Ollama(
        model="llama3.2",  # or "llama3.1", "mistral", etc.
        temperature=0
    )
    
    # Create RAG chain using LCEL
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever


def query_documents(rag_chain, retriever, question):
    """Query the documents and return answer with sources"""
    
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")
    
    # Get answer
    print("Generating answer...")
    answer = rag_chain.invoke(question)
    
    # Get source documents
    source_docs = retriever.invoke(question)
    
    print(f"Answer:\n{answer}\n")
    
    if source_docs:
        print(f"\n{'─'*60}")
        print("Sources:")
        print(f"{'─'*60}")
        
        for i, doc in enumerate(source_docs, 1):
            print(f"\n[{i}] Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Content preview: {doc.page_content[:200]}...")
            print()
    
    return answer, source_docs


def main():
    """Main retrieval pipeline"""
    
    # Load vector store
    vectorstore = load_vector_store(persist_directory="chroma_db")
    
    # Create RAG chain
    rag_chain, retriever = create_rag_chain(vectorstore, k=4)
    
    print("\nRAG system is ready! Ask questions about your documents.")
    print("(Using Ollama - FREE local LLM)")
    print("Type 'quit' or 'exit' to stop.\n")
    
    # Interactive query loop
    while True:
        question = input("Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        try:
            query_documents(rag_chain, retriever, question)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()