from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
import os

class RAGEngine:
    def __init__(self, model_path: str, embedding_model_name: str):
        self.model_path = model_path
        self.embedding_model_name = embedding_model_name
        self.qa_chain = None

    def process_file(self, file_path: str):
        loader = TextLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        db = FAISS.from_documents(docs, embeddings)
        retriever = db.as_retriever()

        llm = LlamaCpp(
            model_path=self.model_path,
            temperature=0.7,
            max_tokens=512,
            n_ctx=2048,
            top_p=1,
            n_threads=os.cpu_count(),
            verbose=False
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

    def query(self, question: str) -> str:
        if self.qa_chain is None:
            raise ValueError("Engine not initialized. Upload and process a file first.")
        result = self.qa_chain({"query": question})
        return result["result"]