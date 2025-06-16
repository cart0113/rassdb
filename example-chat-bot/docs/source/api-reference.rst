API Reference
=============

This document provides detailed information about all API endpoints available in the RASSDB Chat Bot.

Base URLs
---------

* **Frontend Server**: ``http://localhost:3000``
* **Backend Server**: ``http://localhost:8000``

Frontend API Endpoints
----------------------

The frontend server provides these endpoints for the web UI:

Health Check
~~~~~~~~~~~~

.. code-block:: http

   GET /api/health

Check the health status of both frontend and backend servers.

**Response:**

.. code-block:: json

   {
     "status": "healthy",
     "frontend": {
       "version": "0.1.0",
       "uptime": 3600.5
     },
     "backend": {
       "status": "healthy",
       "version": "0.1.0",
       "mcp_initialized": true
     }
   }

Session Management
~~~~~~~~~~~~~~~~~~

.. code-block:: http

   POST /api/session

Create or retrieve a chat session.

**Response:**

.. code-block:: json

   {
     "session_id": "550e8400-e29b-41d4-a716-446655440000",
     "frontend_session_id": "s:abcd1234..."
   }

.. code-block:: http

   DELETE /api/session

Clear the current session and its history.

**Response:**

.. code-block:: json

   {
     "message": "Session cleared successfully"
   }

Query Execution
~~~~~~~~~~~~~~~

.. code-block:: http

   POST /api/query

Execute a query against RASSDB.

**Request Body:**

.. code-block:: json

   {
     "query": "How does authentication work?",
     "top_k": 5,
     "filters": {
       "file_type": "python",
       "directory": "src/"
     }
   }

**Response:**

.. code-block:: json

   {
     "results": [
       {
         "id": "doc_0",
         "content": "Authentication implementation...",
         "score": 0.95,
         "metadata": {
           "file_path": "src/auth.py",
           "line_number": 42,
           "type": "function"
         }
       }
     ],
     "total": 3,
     "query_time": 0.123,
     "session_id": "550e8400-e29b-41d4-a716-446655440000"
   }

Chat History
~~~~~~~~~~~~

.. code-block:: http

   GET /api/history

Retrieve the chat history for the current session.

**Response:**

.. code-block:: json

   {
     "messages": [
       {
         "id": "msg_1",
         "content": "How does authentication work?",
         "role": "user",
         "timestamp": "2024-01-15T10:30:00Z",
         "metadata": null
       },
       {
         "id": "msg_2",
         "content": "I found 3 relevant results...",
         "role": "assistant",
         "timestamp": "2024-01-15T10:30:01Z",
         "metadata": {
           "results_count": 3
         }
       }
     ],
     "session_id": "550e8400-e29b-41d4-a716-446655440000"
   }

Backend API Endpoints
---------------------

The backend server provides the core RASSDB functionality:

Health Check
~~~~~~~~~~~~

.. code-block:: http

   GET /health

**Response Model:**

.. code-block:: python

   class HealthResponse(BaseModel):
       status: str
       version: str
       mcp_initialized: bool

Query Endpoint
~~~~~~~~~~~~~~

.. code-block:: http

   POST /query

**Request Model:**

.. code-block:: python

   class QueryRequest(BaseModel):
       query: str
       session_id: Optional[str] = None
       top_k: int = Field(5, ge=1, le=100)
       filters: Optional[dict] = None

**Response Model:**

.. code-block:: python

   class QueryResponse(BaseModel):
       results: List[dict]
       total: int
       query_time: float
       session_id: Optional[str]

Session Endpoints
~~~~~~~~~~~~~~~~~

.. code-block:: http

   POST /sessions

Create a new chat session.

**Request Body (Optional):**

.. code-block:: json

   {
     "metadata": {
       "user_id": "user123",
       "client": "web"
     }
   }

.. code-block:: http

   GET /sessions/{session_id}/history

Get message history for a specific session.

.. code-block:: http

   DELETE /sessions/{session_id}

Delete a session and its associated data.

Code Search
~~~~~~~~~~~

.. code-block:: http

   POST /similar-code

Find code snippets similar to the provided example.

**Query Parameters:**

* ``code_snippet`` (string, required): The code to find similarities for
* ``language`` (string, optional): Programming language (default: "python")
* ``top_k`` (integer, optional): Number of results (default: 5)

**Response:**

.. code-block:: json

   {
     "results": [
       {
         "id": "similar_0",
         "code": "def authenticate_user(username, password):\n    ...",
         "similarity": 0.92,
         "file_path": "src/auth/validators.py",
         "language": "python"
       }
     ]
   }

Admin Endpoints
~~~~~~~~~~~~~~~

.. code-block:: http

   POST /index

Index a directory for RASSDB searching.

**Request Model:**

.. code-block:: python

   class IndexRequest(BaseModel):
       directory: str
       ignore_patterns: Optional[List[str]] = None

**Response:**

.. code-block:: json

   {
     "message": "Indexing completed",
     "stats": {
       "files_processed": 42,
       "files_ignored": 8,
       "total_size_bytes": 5242880,
       "index_time_seconds": 2.3,
       "embedding_model": "text-embedding-ada-002"
     }
   }

Error Responses
---------------

All endpoints follow a consistent error response format:

.. code-block:: json

   {
     "error": "Error message",
     "details": "Detailed error information",
     "status_code": 400
   }

Common HTTP status codes:

* ``400`` - Bad Request (invalid input)
* ``404`` - Not Found (resource doesn't exist)
* ``500`` - Internal Server Error
* ``503`` - Service Unavailable (backend connection issues)

Rate Limiting
-------------

The API implements rate limiting to prevent abuse:

* Default: 100 requests per minute per IP
* Query endpoint: 30 requests per minute
* Index endpoint: 5 requests per hour

Rate limit headers:

* ``X-RateLimit-Limit``: Maximum requests allowed
* ``X-RateLimit-Remaining``: Requests remaining
* ``X-RateLimit-Reset``: Unix timestamp when limit resets

Authentication
--------------

Currently, the API uses session-based authentication:

1. Session created automatically on first request
2. Session ID stored in HTTP session cookie
3. All subsequent requests use the same session

For production deployments, consider adding:

* API key authentication
* OAuth 2.0 support
* JWT tokens for stateless auth

WebSocket Support (Future)
--------------------------

Planned WebSocket endpoint for real-time updates:

.. code-block:: http

   WS /ws

Features:
* Real-time query responses
* Live indexing progress
* Server status updates