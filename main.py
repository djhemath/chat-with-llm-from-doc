from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag_engine import RAGEngine
import os

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize RAG Engine with model paths
engine = RAGEngine(
    model_path="mistral-7b-instruct-v0.1.Q8_0.gguf",
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
)

app = FastAPI()

# Serve the React app's static files
frontend_build_dir = os.path.join(os.getcwd(), "static")

# Serve React files
app.mount("/static", StaticFiles(directory=os.path.join(frontend_build_dir)), name="static")

class QueryModel(BaseModel):
    question: str

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        engine.process_file(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    return {"message": f"✅ File '{file.filename}' uploaded and processed."}

@app.post("/query")
async def ask(query: QueryModel):
    try:
        answer = engine.query(query.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"❌ Error: {str(e)}")

app.mount("/", StaticFiles(directory=frontend_build_dir, html=True), name="react")