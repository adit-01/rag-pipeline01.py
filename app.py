import gradio as gr
from retrieval_ollama import load_vector_store, create_rag_chain

# Load once at startup
vectorstore = load_vector_store()
rag_chain, retriever = create_rag_chain(vectorstore)

def chat(message, history):
    answer = rag_chain.invoke(message)
    return answer

demo = gr.ChatInterface(
    chat,
    title="Document Q&A Assistant",
    description="Ask questions about your documents"
)

demo.launch()