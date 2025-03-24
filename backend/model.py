from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document as LCDocument
from langchain_core.prompts import ChatPromptTemplate

class State(TypedDict):
    question: str
    context: List[LCDocument]
    answer: str

class RAGSystemGroq:
    def __init__(self, model_name: str, temperature: float = 0.7):
        self.llm = ChatGroq(model_name=model_name, temperature=temperature)
        system_prompt = (
            "Use the following context to answer the question: {context}\n"
            "Question: {input}"
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        self.vector_store = None

    def create_vector_store(self, texts: list[str]) -> FAISS:
        """
        Create a vector store (FAISS) using HuggingFace embeddings.
        """
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
        vector_store = FAISS.from_texts(texts, embeddings)
        self.vector_store = vector_store
        return vector_store

    def build_state_graph(self):
        """
        Build and compile a state graph that defines the RAG pipeline.
        It defines a retrieval step and a generation step.
        """
        def retrieve(state: State):
            retrieved_docs = self.vector_store.similarity_search(state["question"])
            return {"context": retrieved_docs}

        def generate(state: State):
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = self.prompt.invoke({"input": state["question"], "context": docs_content})
            response = self.llm.invoke(messages)
            return {"answer": response.content}

        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        return graph_builder.compile()