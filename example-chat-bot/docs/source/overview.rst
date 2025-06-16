Overview
========

RASSDB Chat Bot is a demonstration project that shows how to build a conversational interface
for querying codebases using RASSDB's vector search capabilities through a Model Context Protocol (MCP) layer.

What is RASSDB?
---------------

RASSDB is a vector database optimized for code search and retrieval. It allows you to:

* Index source code with semantic understanding
* Search using natural language queries
* Find similar code patterns across your codebase
* Maintain context across queries

What is MCP?
------------

The Model Context Protocol (MCP) is an abstraction layer that:

* Manages conversation context and session state
* Translates natural language queries into vector searches
* Formats search results for conversational responses
* Handles the complexity of multi-turn interactions

Architecture Overview
---------------------

The project consists of three main components:

1. **Python Backend Server** (Port 8000)
   
   * FastAPI application
   * Implements the RASSDB MCP handler
   * Manages chat sessions and query processing
   * Provides RESTful API endpoints

2. **JavaScript Frontend Server** (Port 3000)
   
   * Express.js application
   * Serves the web interface
   * Proxies requests to the Python backend
   * Handles session management

3. **Web UI**
   
   * Interactive chat interface
   * Real-time server status monitoring
   * Session history display
   * Query configuration options

Data Flow
---------

.. code-block:: text

   User -> Web UI -> JS Server -> Python Server -> RASSDB
                                       |
                                    MCP Layer
                                       |
                                 Context Management

1. User enters a query in the web interface
2. JavaScript server receives the request and forwards it to Python backend
3. Python server processes the query through the MCP handler
4. MCP handler queries RASSDB and formats results
5. Response flows back through the chain to the user

Use Cases
---------

This architecture is suitable for:

* **Code Documentation Assistants**: Help developers understand unfamiliar codebases
* **Code Search Tools**: Find examples and patterns in large projects
* **Development Chatbots**: Answer questions about project structure and implementation
* **Knowledge Base Systems**: Extract insights from code repositories

Key Benefits
------------

* **Natural Language Interface**: No need to learn query syntax
* **Context Preservation**: Maintains conversation history
* **Scalable Architecture**: Separate frontend and backend services
* **Language Agnostic**: Can index and search any programming language
* **Extensible Design**: Easy to add new features and integrations