MCP Protocol
============

The Model Context Protocol (MCP) is the core abstraction layer that enables intelligent
interaction between natural language queries and the RASSDB vector database.

Overview
--------

MCP serves as a bridge between conversational interfaces and code search functionality:

.. code-block:: text

   Natural Language → MCP Handler → Vector Search → Contextual Response
         Query         (Context)      (RASSDB)       (Formatted)

Key Responsibilities
--------------------

1. **Query Understanding**: Parse and interpret natural language queries
2. **Context Management**: Maintain conversation state across interactions
3. **Search Optimization**: Convert queries into effective vector searches
4. **Response Generation**: Format search results into conversational responses

Core Components
---------------

MCP Handler Class
~~~~~~~~~~~~~~~~~

The ``RASSDBMCPHandler`` class is the main implementation:

.. code-block:: python

   class RASSDBMCPHandler:
       def __init__(self, db_path: Optional[Path] = None, 
                    index_path: Optional[Path] = None):
           """Initialize the MCP handler with database paths."""
           
       async def initialize(self) -> None:
           """Initialize RASSDB connection and load indices."""
           
       async def query(self, query_text: str, top_k: int = 5, 
                      filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
           """Execute a query against RASSDB."""
           
       async def add_context(self, session_id: str, 
                           context: Dict[str, Any]) -> None:
           """Add context to the MCP session."""
           
       async def get_similar_code(self, code_snippet: str, 
                                language: str = "python", 
                                top_k: int = 5) -> List[Dict[str, Any]]:
           """Find similar code snippets in the database."""

Context Management
~~~~~~~~~~~~~~~~~~

MCP maintains context at multiple levels:

1. **Session Context**: User-specific conversation history
2. **Query Context**: Related information from previous queries
3. **Code Context**: Understanding of the codebase structure

.. code-block:: python

   # Example context structure
   context = {
       "session_id": "uuid",
       "user_preferences": {
           "language": "python",
           "detail_level": "verbose"
       },
       "recent_queries": [
           {"query": "authentication", "timestamp": "..."},
           {"query": "user validation", "timestamp": "..."}
       ],
       "discovered_entities": {
           "functions": ["authenticate_user", "validate_token"],
           "classes": ["AuthManager", "TokenValidator"],
           "modules": ["auth.py", "validators.py"]
       }
   }

Query Processing Pipeline
-------------------------

The MCP follows a structured pipeline for query processing:

1. Query Analysis
~~~~~~~~~~~~~~~~~

.. code-block:: python

   def analyze_query(query_text: str) -> QueryIntent:
       """Extract intent and entities from natural language."""
       # Identify query type: definition, usage, explanation, etc.
       # Extract code entities: function names, classes, concepts
       # Determine scope: file, module, project-wide
       return QueryIntent(
           type="find_usage",
           entities=["authenticate_user"],
           scope="project"
       )

2. Context Enhancement
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def enhance_with_context(query: QueryIntent, 
                          context: SessionContext) -> EnhancedQuery:
       """Augment query with session context."""
       # Add related entities from previous queries
       # Apply user preferences
       # Expand abbreviations and aliases
       return EnhancedQuery(
           original=query,
           expanded_entities=["authenticate_user", "auth_user", "login"],
           filters={"file_type": "python", "exclude": "tests/"}
       )

3. Vector Search Execution
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   async def execute_search(query: EnhancedQuery) -> SearchResults:
       """Perform vector search on RASSDB."""
       # Convert to embedding vector
       # Apply similarity search
       # Post-process with filters
       return SearchResults(
           documents=[...],
           scores=[...],
           metadata=[...]
       )

4. Response Formatting
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def format_response(results: SearchResults, 
                      context: SessionContext) -> FormattedResponse:
       """Convert search results to conversational format."""
       # Rank and filter results
       # Generate natural language summary
       # Include code snippets with highlighting
       # Add follow-up suggestions
       return FormattedResponse(
           summary="Found 3 authentication implementations",
           details=[...],
           suggestions=["Would you like to see the tests?"]
       )

Advanced Features
-----------------

Semantic Understanding
~~~~~~~~~~~~~~~~~~~~~~

MCP includes semantic analysis capabilities:

* **Synonym Recognition**: "auth" → "authentication", "authorize"
* **Concept Mapping**: "user login" → authentication flow
* **Language Detection**: Automatically identify code language
* **Intent Classification**: Question vs. command vs. search

Code-Aware Processing
~~~~~~~~~~~~~~~~~~~~~

Special handling for code-related queries:

.. code-block:: python

   # Recognize code patterns
   if is_code_snippet(query_text):
       return await find_similar_code(query_text)
   
   # Handle specific code queries
   if is_asking_about_function(query_text):
       return await get_function_details(extracted_function_name)
   
   # Process import/dependency queries
   if is_dependency_query(query_text):
       return await analyze_dependencies(query_text)

Multi-Turn Conversations
~~~~~~~~~~~~~~~~~~~~~~~~

MCP maintains conversation flow:

.. code-block:: python

   class ConversationManager:
       def __init__(self):
           self.history: List[Turn] = []
           self.entities: Set[str] = set()
           self.topic: Optional[str] = None
       
       def add_turn(self, query: str, response: str):
           """Record conversation turn and extract context."""
           self.history.append(Turn(query, response))
           self.extract_entities(query, response)
           self.update_topic()
       
       def get_relevant_context(self, new_query: str) -> Dict:
           """Retrieve context relevant to new query."""
           return {
               "previous_entities": list(self.entities),
               "current_topic": self.topic,
               "recent_turns": self.history[-3:]
           }

Extension Points
----------------

The MCP is designed to be extensible:

Custom Filters
~~~~~~~~~~~~~~

.. code-block:: python

   @mcp_handler.register_filter
   def complexity_filter(results: List[Dict]) -> List[Dict]:
       """Filter results by code complexity."""
       return [r for r in results 
               if calculate_complexity(r['content']) < threshold]

Query Preprocessors
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @mcp_handler.register_preprocessor
   def expand_abbreviations(query: str) -> str:
       """Expand common abbreviations."""
       abbreviations = {
           "fn": "function",
           "cls": "class",
           "impl": "implementation"
       }
       for abbr, full in abbreviations.items():
           query = query.replace(abbr, full)
       return query

Response Postprocessors
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @mcp_handler.register_postprocessor
   def add_documentation_links(response: Dict) -> Dict:
       """Add links to relevant documentation."""
       for result in response['results']:
           if doc_url := find_documentation(result['file_path']):
               result['documentation'] = doc_url
       return response

Best Practices
--------------

Query Design
~~~~~~~~~~~~

1. **Be Specific**: Include file names, function names when known
2. **Use Context**: Reference previous results ("show me more like that")
3. **Specify Intent**: "explain", "find usage", "show definition"

Context Management
~~~~~~~~~~~~~~~~~~

1. **Limit History**: Keep only recent relevant turns
2. **Extract Entities**: Build a knowledge graph of discovered code
3. **Track Topics**: Identify when conversation shifts focus

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Cache Results**: Store frequently accessed queries
2. **Batch Operations**: Process multiple related queries together
3. **Lazy Loading**: Defer expensive operations until needed

Error Handling
~~~~~~~~~~~~~~

1. **Graceful Degradation**: Fall back to keyword search if vector search fails
2. **User Feedback**: Request clarification for ambiguous queries
3. **Logging**: Track query patterns for improvement

Future Enhancements
-------------------

Planned improvements to the MCP:

* **Multi-modal Support**: Handle diagrams and documentation
* **Code Generation**: Suggest code completions based on context
* **Refactoring Assistance**: Identify improvement opportunities
* **Cross-Repository Search**: Query multiple codebases simultaneously
* **Learning Capabilities**: Improve from user feedback