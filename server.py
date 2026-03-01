import time
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.agent_controller import run_agent

try:
    from memory.memory_api import search_memory, get_document_count
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    def search_memory(query: str) -> str:
        return ""
    def get_document_count() -> int:
        return 0

app = FastAPI()


# ── Serve app.html at root ─────────────────────────────────────────────────────
@app.get("/")
def serve_ui():
    return FileResponse("app.html")


@app.get("/health")
def health():
    return {"status": "ok", "memory_available": MEMORY_AVAILABLE}


@app.get("/memory_status")
def memory_status():
    return {"documents_indexed": get_document_count()}


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(req: ChatRequest):
    start = time.time()

    memory_context = search_memory(req.message) if MEMORY_AVAILABLE else ""

    result = run_agent(message=req.message, memory_context=memory_context)

    elapsed_ms = round((time.time() - start) * 1000)

    return {
        "response":         result["reply"],
        "execution_log":    "\n".join(result["logs"]),
        "inference_time_ms": elapsed_ms,
        "file_ready":       result["file_ready"],
        "file_path":        result["file_path"],
    }

 
@app.get("/download")
def download_file(path: str):
    if not path or not os.path.exists(path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=os.path.basename(path),
    )