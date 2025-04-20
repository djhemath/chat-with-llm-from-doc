from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
import os

# Constants
DATA_FILE = "data.txt"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_PATH = "mistral-7b-instruct-v0.1.Q8_0.gguf"  # You must download this manually

# Load and split the text
loader = TextLoader(DATA_FILE)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Embeddings + FAISS Vector Store
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()

# Load local LLM using llama-cpp
llm = LlamaCpp(
    model_path=LLM_MODEL_PATH,
    temperature=0.7,
    max_tokens=512,
    n_ctx=2048,
    top_p=1,
    n_threads=os.cpu_count(),
    verbose=False
)

# Setup RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Interactive Q&A
print("📄 Document loaded. Ask questions about it (type 'exit' to quit):")
while True:
    query = input("\n❓ You: ")
    if query.lower() in ("exit", "quit"):
        break

    result = qa_chain({"query": query})
    print("\n🤖 Answer:", result["result"])