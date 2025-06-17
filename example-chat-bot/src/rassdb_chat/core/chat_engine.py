"""Chat engine that integrates with RASSDB MCP."""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import List, Optional

from ..models.chat import ChatSession, Message, QueryRequest, QueryResponse
from .mcp import RASSDBMCPHandler


class ChatEngine:
    """Main chat engine that processes queries through RASSDB MCP."""
    
    def __init__(self, mcp_handler: RASSDBMCPHandler):
        """Initialize the chat engine.
        
        Args:
            mcp_handler: RASSDB MCP handler instance
        """
        self.mcp_handler = mcp_handler
        self.sessions: dict[str, ChatSession] = {}
    
    async def create_session(self, metadata: Optional[dict] = None) -> ChatSession:
        """Create a new chat session.
        
        Args:
            metadata: Optional session metadata
            
        Returns:
            New chat session
        """
        session = ChatSession(
            id=str(uuid.uuid4()),
            metadata=metadata or {}
        )
        self.sessions[session.id] = session
        return session
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process a user query through RASSDB MCP.
        
        Args:
            request: Query request
            
        Returns:
            Query response with results
        """
        start_time = asyncio.get_event_loop().time()
        
        # Get or create session
        if request.session_id and request.session_id in self.sessions:
            session = self.sessions[request.session_id]
        else:
            session = await self.create_session()
            request.session_id = session.id
        
        # Add user message to session
        user_message = Message(
            id=str(uuid.uuid4()),
            content=request.query,
            role="user"
        )
        session.messages.append(user_message)
        
        # Query RASSDB through MCP with timeout
        try:
            results = await asyncio.wait_for(
                self.mcp_handler.query(
                    query_text=request.query,
                    top_k=request.top_k,
                    filters=request.filters
                ),
                timeout=30.0  # 30 second timeout
            )
        except asyncio.TimeoutError:
            logging.error(f"[ChatEngine] Query timed out after 30 seconds")
            results = []
        
        # Generate assistant response based on results
        if results:
            response_content = self._format_results(results)
        else:
            response_content = "I couldn't find any relevant information for your query."
        
        # Add assistant message to session
        assistant_message = Message(
            id=str(uuid.uuid4()),
            content=response_content,
            role="assistant",
            metadata={"results_count": len(results)}
        )
        session.messages.append(assistant_message)
        
        # Update session timestamp
        session.updated_at = datetime.utcnow()
        
        # Calculate query time
        query_time = asyncio.get_event_loop().time() - start_time
        
        return QueryResponse(
            results=results,
            total=len(results),
            query_time=query_time,
            session_id=session.id
        )
    
    def _format_results(self, results: List[dict]) -> str:
        """Format query results into a readable response.
        
        Args:
            results: List of query results
            
        Returns:
            Formatted response string
        """
        if not results:
            return "No results found."
        
        response = f"I found {len(results)} relevant result(s):\n\n"
        
        for i, result in enumerate(results, 1):
            response += f"{i}. **{result.get('metadata', {}).get('file_path', 'Unknown file')}**"
            if 'line_number' in result.get('metadata', {}):
                response += f" (line {result['metadata']['line_number']})"
            response += f"\n   Score: {result.get('score', 0):.2f}\n"
            response += f"   {result.get('content', 'No content available')}\n\n"
        
        return response.strip()
    
    async def get_session_history(self, session_id: str) -> Optional[List[Message]]:
        """Get message history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of messages if session exists
        """
        if session_id in self.sessions:
            return self.sessions[session_id].messages
        return None
    
    async def clear_session(self, session_id: str) -> bool:
        """Clear a chat session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was cleared
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            await self.mcp_handler.clear_context(session_id)
            return True
        return False