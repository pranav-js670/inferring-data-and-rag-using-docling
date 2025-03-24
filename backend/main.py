import os
from dotenv import load_dotenv
load_dotenv()

from services.extraction import DoclingPDFLoader
from model import RAGSystemGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

def main():
    pdf_path = "report.pdf"
    print("Starting PDF loading...")
    loader = DoclingPDFLoader(file_path=pdf_path)
    docs = list(loader.lazy_load())
    print(f"Loaded {len(docs)} document(s).")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    print(f"Split documents into {len(chunks)} chunks.")
    
    rag_system = RAGSystemGroq(model_name="llama-3.1-8b-instant", temperature=0.3)
    print("Creating vector store...")
    vector_store = rag_system.create_vector_store([chunk.page_content for chunk in chunks])
    print("Vector store created.")
    
    qa_graph = rag_system.build_state_graph()
    print("State graph built.")
    
    initial_state = {"question": "What is the name of the patient and summarize the history and also state treatments available for the diagnosis??", "context": [], "answer": ""}
    print("Running QA pipeline...")
    final_state = qa_graph.invoke(initial_state)
    print("\nAnswer:\n", final_state["answer"])

    # Run the state graph to get the final answer
    # final_state = qa_graph.invoke(initial_state)
    # print("\nAnswer:\n", final_state["answer"])

if __name__ == "__main__":
    main()
