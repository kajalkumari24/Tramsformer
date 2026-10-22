"""FastAPI backend for the Mini GPT chatbot.

Run from the project folder:
    python -m uvicorn backend.app:create_app --factory --port 8000
Then open http://127.0.0.1:8000  (the chat page is served by this same app).
"""
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = {"max_new_tokens": 200, "temperature": 0.8, "top_k": 40}
LIMITS = {"max_new_tokens": [1, 500], "temperature": [0.0, 2.0]}


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class Settings(BaseModel):
    max_new_tokens: int = Field(DEFAULTS["max_new_tokens"], ge=1, le=500)
    temperature: float = Field(DEFAULTS["temperature"], ge=0.0, le=2.0)   # 0 = greedy
    top_k: Optional[int] = Field(DEFAULTS["top_k"], ge=1, le=1000)         # None = off
    seed: Optional[int] = Field(None, ge=0, le=2**31 - 1)                  # None = random


class GenerateRequest(Settings):
    prompt: str = Field(min_length=1, max_length=4000)


class ChatRequest(Settings):
    messages: List[Message] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def last_message_must_be_from_the_user(self):
        if self.messages[-1].role != "user":
            raise ValueError("The last message must come from the user")
        return self


def create_app(service=None, frontend_dir=None, checkpoint_path=None, tokenizer_path=None):
    """Build the app. Pass `service` to inject a model (used by the tests)."""
    checkpoint_path = Path(checkpoint_path or PROJECT_ROOT / "checkpoints" / "best.pt")
    tokenizer_path = Path(tokenizer_path or PROJECT_ROOT / "checkpoints" / "tokenizer.json")
    frontend_dir = Path(frontend_dir or PROJECT_ROOT / "frontend")

    state = {"service": service}
    load_lock = threading.Lock()

    def get_service():
        if state["service"] is None:
            with load_lock:
                if state["service"] is None:
                    if not checkpoint_path.is_file():
                        raise HTTPException(
                            status_code=503,
                            detail=f"No trained model found at {checkpoint_path}. Train it first (Step 6).",
                        )
                    from backend.service import ModelService   # imports PyTorch, so only when needed
                    state["service"] = ModelService.from_checkpoint(checkpoint_path, tokenizer_path)
        return state["service"]

    @asynccontextmanager
    async def lifespan(app):
        try:
            get_service()                 # load the model now so the first request is fast
        except Exception as error:
            print(f"[backend] Model not loaded at startup: {error}")
        yield

    app = FastAPI(title="Mini GPT API", version="1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/api/health")
    def health():
        return {"status": "ok", "model_loaded": state["service"] is not None}

    @app.get("/api/info")
    def info():
        model_info = get_service().info()
        limits = {**LIMITS, "top_k": [1, int(model_info.get("vocab_size", 1000))]}
        return {**model_info, "defaults": DEFAULTS, "limits": limits}

    @app.post("/api/generate")
    def generate_text(request: GenerateRequest):
        return get_service().complete(
            request.prompt, request.max_new_tokens, request.temperature, request.top_k, request.seed
        )

    @app.post("/api/chat")
    def chat(request: ChatRequest):
        messages = [m.model_dump() for m in request.messages]
        return get_service().chat(
            messages, request.max_new_tokens, request.temperature, request.top_k, request.seed
        )

    # The chat page. Mounted last so the /api routes above take priority.
    if (frontend_dir / "index.html").is_file():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

    return app
