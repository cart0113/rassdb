"""Chat-related data models."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Represents a chat message."""
    
    id: str = Field(..., description="Unique message identifier")
    content: str = Field(..., description="Message content")
    role: str = Field(..., description="Message role (user/assistant)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    metadata: Optional[dict] = Field(default=None, description="Additional metadata")


class ChatSession(BaseModel):
    """Represents a chat session."""
    
    id: str = Field(..., description="Unique session identifier")
    messages: List[Message] = Field(default_factory=list, description="List of messages in the session")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Session creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    metadata: Optional[dict] = Field(default=None, description="Session metadata")


class QueryRequest(BaseModel):
    """Request model for querying RASSDB."""
    
    query: str = Field(..., description="Query string")
    session_id: Optional[str] = Field(None, description="Session ID for context")
    top_k: int = Field(5, description="Number of results to return", ge=1, le=100)
    filters: Optional[dict] = Field(None, description="Query filters")


class QueryResponse(BaseModel):
    """Response model for RASSDB queries."""
    
    results: List[dict] = Field(..., description="Query results")
    total: int = Field(..., description="Total number of results")
    query_time: float = Field(..., description="Query execution time in seconds")
    session_id: Optional[str] = Field(None, description="Session ID if provided")