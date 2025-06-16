"""FastAPI server for RASSDB Chat MCP layer."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..core.chat_engine import ChatEngine
from ..core.mcp import RASSDBMCPHandler
from ..models.chat import QueryRequest, QueryResponse, ChatSession, Message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    mcp_initialized: bool


class IndexRequest(BaseModel):
    """Request model for indexing operations."""
    directory: str
    ignore_patterns: Optional[list[str]] = None


# Global instances
mcp_handler: Optional[RASSDBMCPHandler] = None
chat_engine: Optional[ChatEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global mcp_handler, chat_engine
    
    # Startup
    logger.info("Starting RASSDB Chat Server...")
    mcp_handler = RASSDBMCPHandler()
    await mcp_handler.initialize()
    chat_engine = ChatEngine(mcp_handler)
    logger.info("Server initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down server...")


# Create FastAPI app
app = FastAPI(
    title="RASSDB Chat API",
    description="RESTful API for RASSDB Model Context Protocol",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server health status."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        mcp_initialized=mcp_handler.is_initialized if mcp_handler else False
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Execute a query against RASSDB."""
    if not chat_engine:
        raise HTTPException(status_code=500, detail="Chat engine not initialized")
    
    try:
        response = await chat_engine.process_query(request)
        return response
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions", response_model=ChatSession)
async def create_session(metadata: Optional[dict] = None):
    """Create a new chat session."""
    if not chat_engine:
        raise HTTPException(status_code=500, detail="Chat engine not initialized")
    
    session = await chat_engine.create_session(metadata)
    return session


@app.get("/sessions/{session_id}/history", response_model=list[Message])
async def get_session_history(session_id: str):
    """Get message history for a session."""
    if not chat_engine:
        raise HTTPException(status_code=500, detail="Chat engine not initialized")
    
    history = await chat_engine.get_session_history(session_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return history


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a chat session."""
    if not chat_engine:
        raise HTTPException(status_code=500, detail="Chat engine not initialized")
    
    success = await chat_engine.clear_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"message": "Session cleared successfully"}


@app.post("/index")
async def index_directory(request: IndexRequest):
    """Index a directory for RASSDB."""
    if not mcp_handler:
        raise HTTPException(status_code=500, detail="MCP handler not initialized")
    
    try:
        directory = Path(request.directory)
        if not directory.exists():
            raise HTTPException(status_code=400, detail="Directory does not exist")
        
        stats = await mcp_handler.index_directory(directory, request.ignore_patterns)
        return {"message": "Indexing completed", "stats": stats}
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similar-code")
async def find_similar_code(code_snippet: str, language: str = "python", top_k: int = 5):
    """Find similar code snippets."""
    if not mcp_handler:
        raise HTTPException(status_code=500, detail="MCP handler not initialized")
    
    try:
        results = await mcp_handler.get_similar_code(code_snippet, language, top_k)
        return {"results": results}
    except Exception as e:
        logger.error(f"Similar code search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Main entry point for the server."""
    uvicorn.run(
        "rassdb_chat.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()