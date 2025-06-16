Architecture
============

This document provides a detailed overview of the RASSDB Chat Bot architecture,
including component interactions, data flow, and design decisions.

System Components
-----------------

The application follows a three-tier architecture:

.. code-block:: text

   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
   │                 │     │                 │     │                 │
   │   Web Browser   │────▶│ Frontend Server │────▶│ Backend Server  │
   │   (Client UI)   │◀────│   (Node.js)     │◀────│   (Python)      │
   │                 │     │                 │     │                 │
   └─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                             │
                                                             ▼
                                                    ┌─────────────────┐
                                                    │                 │
                                                    │     RASSDB      │
                                                    │   (Database)    │
                                                    │                 │
                                                    └─────────────────┘

Component Details
-----------------

Web Browser (Client UI)
~~~~~~~~~~~~~~~~~~~~~~~

The client-side application provides:

* **Interactive Chat Interface**: Real-time message display
* **Query Input**: Natural language question input
* **Session Management**: Client-side session tracking
* **Status Monitoring**: Server health visualization

Technologies:

* Vanilla JavaScript (no framework dependencies)
* HTML5 and CSS3
* Fetch API for HTTP requests
* Local Storage for client preferences

Frontend Server (Node.js)
~~~~~~~~~~~~~~~~~~~~~~~~~

The Express.js server acts as a middleware layer:

* **Static File Serving**: Hosts the web UI assets
* **API Gateway**: Routes requests to the backend
* **Session Management**: Maintains HTTP sessions
* **Request Validation**: Ensures proper request format

Key Features:

* CORS support for cross-origin requests
* Request logging with Morgan
* Environment-based configuration
* Graceful error handling

Backend Server (Python)
~~~~~~~~~~~~~~~~~~~~~~~

The FastAPI server implements the core logic:

* **MCP Implementation**: Handles the Model Context Protocol
* **RASSDB Integration**: Manages vector database queries
* **Chat Engine**: Processes conversations and maintains context
* **API Endpoints**: RESTful interface for all operations

Core Components:

1. **MCP Handler** (``core/mcp.py``)
   
   * Manages RASSDB connections
   * Executes vector searches
   * Maintains query context
   * Handles similar code searches

2. **Chat Engine** (``core/chat_engine.py``)
   
   * Processes user queries
   * Manages chat sessions
   * Formats responses
   * Tracks conversation history

3. **API Server** (``api/server.py``)
   
   * FastAPI application setup
   * Endpoint definitions
   * Request/response handling
   * Lifecycle management

Data Models
-----------

The application uses Pydantic models for data validation:

Message Model
~~~~~~~~~~~~~

.. code-block:: python

   class Message(BaseModel):
       id: str
       content: str
       role: str  # "user" or "assistant"
       timestamp: datetime
       metadata: Optional[dict]

Chat Session Model
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class ChatSession(BaseModel):
       id: str
       messages: List[Message]
       created_at: datetime
       updated_at: datetime
       metadata: Optional[dict]

Query Models
~~~~~~~~~~~~

.. code-block:: python

   class QueryRequest(BaseModel):
       query: str
       session_id: Optional[str]
       top_k: int = 5
       filters: Optional[dict]

   class QueryResponse(BaseModel):
       results: List[dict]
       total: int
       query_time: float
       session_id: Optional[str]

API Design
----------

The API follows RESTful principles:

Core Endpoints
~~~~~~~~~~~~~~

* ``GET /health`` - System health check
* ``POST /query`` - Execute a query
* ``POST /sessions`` - Create a new session
* ``GET /sessions/{id}/history`` - Get session history
* ``DELETE /sessions/{id}`` - Clear a session
* ``POST /index`` - Index a directory
* ``POST /similar-code`` - Find similar code

Request Flow
~~~~~~~~~~~~

1. **Client Request**: User submits query through web UI
2. **Frontend Validation**: Basic input validation
3. **Backend Processing**: Query analysis and RASSDB search
4. **Response Formation**: Results formatted for display
5. **Client Update**: UI updated with response

Security Considerations
-----------------------

Session Management
~~~~~~~~~~~~~~~~~~

* Server-side session storage
* Session IDs generated with cryptographic randomness
* Configurable session timeout
* CORS configured for specific origins

Input Validation
~~~~~~~~~~~~~~~~

* Pydantic models validate all inputs
* SQL injection prevention through parameterized queries
* XSS protection through proper escaping
* Rate limiting on API endpoints (configurable)

Performance Optimization
------------------------

Caching Strategy
~~~~~~~~~~~~~~~~

* In-memory context cache for active sessions
* Query result caching (configurable TTL)
* Static asset caching headers
* Database connection pooling

Asynchronous Processing
~~~~~~~~~~~~~~~~~~~~~~~

* Async/await throughout the Python backend
* Non-blocking I/O for database queries
* Concurrent request handling
* Background task support for indexing

Scalability Considerations
--------------------------

Horizontal Scaling
~~~~~~~~~~~~~~~~~~

* Stateless API design enables load balancing
* Session storage can be externalized (Redis)
* Database connections support pooling
* Frontend and backend can scale independently

Vertical Scaling
~~~~~~~~~~~~~~~~

* Async architecture maximizes CPU utilization
* Memory-efficient data structures
* Streaming responses for large results
* Configurable worker processes

Monitoring and Observability
----------------------------

Logging
~~~~~~~

* Structured logging with context
* Log levels: DEBUG, INFO, WARNING, ERROR
* Request/response logging
* Performance metrics logging

Health Checks
~~~~~~~~~~~~~

* Liveness probe: ``/health``
* Readiness probe: Database connectivity
* Dependency health aggregation
* Metric endpoints (Prometheus-compatible)

Error Handling
~~~~~~~~~~~~~~

* Graceful degradation
* User-friendly error messages
* Detailed error logging
* Automatic retry logic for transient failures