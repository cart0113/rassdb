RASSDB Chat Bot Documentation
=============================

Welcome to the RASSDB Chat Bot documentation! This project demonstrates how to build a modern
chat interface that integrates with RASSDB through a Model Context Protocol (MCP) layer.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   getting-started
   architecture
   api-reference
   mcp-protocol
   deployment
   development
   troubleshooting

Overview
--------

RASSDB Chat Bot is a full-stack application that showcases:

* **Python Backend**: FastAPI server implementing the RASSDB MCP layer
* **JavaScript Frontend**: Node.js/Express server with interactive web UI
* **RESTful API**: Clean JSON-based communication between frontend and backend
* **Session Management**: Persistent chat sessions with history
* **Code Search**: Natural language queries against your codebase

Key Features
------------

* **Natural Language Queries**: Ask questions about your codebase in plain English
* **Context-Aware Responses**: The MCP layer maintains conversation context
* **Similar Code Search**: Find code snippets similar to what you're looking for
* **Real-time Status**: Monitor server health and connection status
* **Session Persistence**: Continue conversations across page refreshes

Quick Example
-------------

.. code-block:: javascript

   // Frontend query
   POST /api/query
   {
     "query": "How does the authentication system work?",
     "top_k": 5
   }

.. code-block:: python

   # Backend processing
   async def process_query(request: QueryRequest) -> QueryResponse:
       results = await mcp_handler.query(
           query_text=request.query,
           top_k=request.top_k
       )
       return QueryResponse(results=results, ...)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`