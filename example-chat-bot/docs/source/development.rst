Development Guide
=================

This guide covers setting up a development environment and contributing to the RASSDB Chat Bot project.

Development Setup
-----------------

Prerequisites
~~~~~~~~~~~~~

Ensure you have the following tools installed:

* Python 3.8+ with pip and venv
* Node.js 14+ with npm
* Git
* A code editor (VSCode recommended)
* Docker (optional, for containerized development)

Setting Up the Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Clone and Fork**:

   .. code-block:: bash

      # Fork the repository on GitHub first
      git clone https://github.com/YOUR_USERNAME/rassdb-chat.git
      cd rassdb-chat
      git remote add upstream https://github.com/example/rassdb-chat.git

2. **Python Backend Setup**:

   .. code-block:: bash

      # Create virtual environment
      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      
      # Install in development mode
      pip install -e ".[dev]"
      
      # Install pre-commit hooks
      pre-commit install

3. **JavaScript Frontend Setup**:

   .. code-block:: bash

      cd src/frontend
      npm install
      
      # Install development dependencies
      npm install --save-dev eslint prettier nodemon

4. **Environment Configuration**:

   .. code-block:: bash

      # Backend development config
      export RASSDB_DEV_MODE=true
      export LOG_LEVEL=DEBUG
      
      # Frontend development config
      cp .env.example .env.development
      # Edit .env.development with your settings

Project Structure
-----------------

Understanding the codebase organization:

.. code-block:: text

   example-chat-bot/
   ├── src/
   │   ├── rassdb_chat/          # Python backend package
   │   │   ├── api/              # FastAPI endpoints
   │   │   │   ├── __init__.py
   │   │   │   └── server.py     # Main API server
   │   │   ├── core/             # Core business logic
   │   │   │   ├── __init__.py
   │   │   │   ├── mcp.py        # MCP implementation
   │   │   │   └── chat_engine.py # Chat processing
   │   │   └── models/           # Pydantic models
   │   │       ├── __init__.py
   │   │       └── chat.py       # Data models
   │   └── frontend/             # JavaScript frontend
   │       ├── src/
   │       │   └── server.js     # Express server
   │       └── public/           # Static assets
   │           ├── index.html
   │           ├── app.js
   │           └── styles.css
   ├── tests/                    # Test suites
   │   ├── test_api.py
   │   ├── test_mcp.py
   │   └── test_frontend.js
   ├── docs/                     # Sphinx documentation
   └── scripts/                  # Utility scripts

Development Workflow
--------------------

Running in Development Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Backend Development Server**:

   .. code-block:: bash

      # Auto-reload on code changes
      uvicorn rassdb_chat.api.server:app --reload --log-level debug
      
      # Or use the make command
      make run-backend

2. **Frontend Development Server**:

   .. code-block:: bash

      cd src/frontend
      npm run dev  # Uses nodemon for auto-reload

3. **Full Stack Development**:

   .. code-block:: bash

      # Use the development script
      ./scripts/dev.sh
      
      # Or use docker-compose
      docker-compose -f docker-compose.dev.yml up

Code Style and Formatting
~~~~~~~~~~~~~~~~~~~~~~~~~

Python Code Style
^^^^^^^^^^^^^^^^^

The project uses Black for formatting and Ruff for linting:

.. code-block:: bash

   # Format code
   black src/rassdb_chat tests
   
   # Run linter
   ruff src/rassdb_chat tests
   
   # Type checking
   mypy src/rassdb_chat

JavaScript Code Style
^^^^^^^^^^^^^^^^^^^^^

The project uses ESLint and Prettier:

.. code-block:: bash

   cd src/frontend
   
   # Lint code
   npm run lint
   
   # Format code
   npm run format

Commit Message Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~

Follow the conventional commits specification:

.. code-block:: text

   feat: add new query filter support
   fix: resolve session timeout issue
   docs: update deployment guide
   test: add MCP handler unit tests
   refactor: simplify chat engine logic
   perf: optimize vector search queries
   chore: update dependencies

Testing
-------

Running Tests
~~~~~~~~~~~~~

Python Tests
^^^^^^^^^^^^

.. code-block:: bash

   # Run all tests
   pytest
   
   # Run with coverage
   pytest --cov=rassdb_chat --cov-report=html
   
   # Run specific test file
   pytest tests/test_api.py
   
   # Run with verbose output
   pytest -v

JavaScript Tests
^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd src/frontend
   
   # Run all tests
   npm test
   
   # Run in watch mode
   npm test -- --watch
   
   # Generate coverage report
   npm test -- --coverage

Writing Tests
~~~~~~~~~~~~~

Python Test Example
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # tests/test_chat_engine.py
   import pytest
   from rassdb_chat.core.chat_engine import ChatEngine
   from rassdb_chat.models.chat import QueryRequest
   
   @pytest.fixture
   async def chat_engine():
       """Create a test chat engine instance."""
       from rassdb_chat.core.mcp import RASSDBMCPHandler
       mcp = RASSDBMCPHandler()
       await mcp.initialize()
       return ChatEngine(mcp)
   
   @pytest.mark.asyncio
   async def test_process_query(chat_engine):
       """Test query processing."""
       request = QueryRequest(
           query="Find authentication functions",
           top_k=3
       )
       response = await chat_engine.process_query(request)
       
       assert response.total >= 0
       assert response.query_time > 0
       assert response.session_id is not None

JavaScript Test Example
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: javascript

   // tests/test_frontend.js
   const request = require('supertest');
   const app = require('../src/server');
   
   describe('API Endpoints', () => {
       test('GET /api/health returns status', async () => {
           const response = await request(app)
               .get('/api/health')
               .expect(200);
           
           expect(response.body).toHaveProperty('status');
           expect(response.body.status).toBe('healthy');
       });
       
       test('POST /api/query processes request', async () => {
           const response = await request(app)
               .post('/api/query')
               .send({ query: 'test query', top_k: 5 })
               .expect(200);
           
           expect(response.body).toHaveProperty('results');
           expect(Array.isArray(response.body.results)).toBe(true);
       });
   });

Debugging
---------

Backend Debugging
~~~~~~~~~~~~~~~~~

1. **VSCode Configuration**:

   .. code-block:: json

      {
          "version": "0.2.0",
          "configurations": [
              {
                  "name": "Python: FastAPI",
                  "type": "python",
                  "request": "launch",
                  "module": "uvicorn",
                  "args": [
                      "rassdb_chat.api.server:app",
                      "--reload"
                  ],
                  "jinja": true,
                  "justMyCode": false
              }
          ]
      }

2. **Debug Logging**:

   .. code-block:: python

      import logging
      logger = logging.getLogger(__name__)
      
      # Add debug statements
      logger.debug(f"Processing query: {query_text}")
      logger.debug(f"Results found: {len(results)}")

Frontend Debugging
~~~~~~~~~~~~~~~~~~

1. **Browser DevTools**: Use Chrome/Firefox developer tools
2. **Node.js Debugging**:

   .. code-block:: json

      {
          "type": "node",
          "request": "launch",
          "name": "Debug Frontend",
          "program": "${workspaceFolder}/src/frontend/src/server.js",
          "envFile": "${workspaceFolder}/src/frontend/.env.development"
      }

API Development
---------------

Adding New Endpoints
~~~~~~~~~~~~~~~~~~~~

1. **Define the Model**:

   .. code-block:: python

      # src/rassdb_chat/models/new_feature.py
      from pydantic import BaseModel
      
      class NewFeatureRequest(BaseModel):
          param1: str
          param2: int
      
      class NewFeatureResponse(BaseModel):
          result: str
          metadata: dict

2. **Implement the Handler**:

   .. code-block:: python

      # src/rassdb_chat/core/new_handler.py
      async def process_new_feature(request: NewFeatureRequest):
          # Implementation logic
          return NewFeatureResponse(...)

3. **Add the Endpoint**:

   .. code-block:: python

      # src/rassdb_chat/api/server.py
      @app.post("/new-feature", response_model=NewFeatureResponse)
      async def new_feature(request: NewFeatureRequest):
          return await process_new_feature(request)

4. **Update Frontend**:

   .. code-block:: javascript

      // src/frontend/src/server.js
      app.post('/api/new-feature', async (req, res) => {
          try {
              const response = await axios.post(
                  `${BACKEND_URL}/new-feature`,
                  req.body
              );
              res.json(response.data);
          } catch (error) {
              res.status(500).json({ error: error.message });
          }
      });

Database Development
--------------------

Working with RASSDB
~~~~~~~~~~~~~~~~~~~

1. **Local RASSDB Setup**:

   .. code-block:: bash

      # Create test database
      mkdir -p .rassdb/test
      
      # Index test data
      rassdb index --path ./test_data \
                  --output .rassdb/test/test.rassdb

2. **Mock RASSDB for Testing**:

   .. code-block:: python

      # tests/mocks/mock_rassdb.py
      class MockRASSDBHandler:
          async def query(self, query_text, top_k=5):
              # Return mock results
              return [
                  {
                      "id": "mock_1",
                      "content": "Mock result",
                      "score": 0.95
                  }
              ]

Documentation Development
-------------------------

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd docs
   
   # Build HTML documentation
   make html
   
   # Auto-rebuild on changes
   sphinx-autobuild source build/html
   
   # Check for broken links
   make linkcheck

Adding Documentation
~~~~~~~~~~~~~~~~~~~~

1. **API Documentation**: Use docstrings with Sphinx format
2. **User Guides**: Add RST files to `docs/source/`
3. **Code Examples**: Include working examples
4. **Diagrams**: Use Mermaid or PlantUML

Performance Profiling
---------------------

Python Profiling
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Profile specific functions
   import cProfile
   import pstats
   
   profiler = cProfile.Profile()
   profiler.enable()
   
   # Code to profile
   await process_query(request)
   
   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(10)

JavaScript Profiling
~~~~~~~~~~~~~~~~~~~~

.. code-block:: javascript

   // Use console.time for simple profiling
   console.time('query-processing');
   const result = await processQuery(query);
   console.timeEnd('query-processing');
   
   // Or use the built-in profiler
   // node --prof src/server.js

Contributing Guidelines
-----------------------

Pull Request Process
~~~~~~~~~~~~~~~~~~~~

1. **Create Feature Branch**:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. **Make Changes**: Follow code style guidelines

3. **Write Tests**: Ensure coverage for new features

4. **Update Documentation**: Add/update relevant docs

5. **Submit PR**: Include clear description and link issues

Code Review Checklist
~~~~~~~~~~~~~~~~~~~~~

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] New features have test coverage
- [ ] Documentation is updated
- [ ] No security vulnerabilities introduced
- [ ] Performance impact considered
- [ ] Breaking changes documented

Release Process
---------------

Version Management
~~~~~~~~~~~~~~~~~~

The project follows semantic versioning:

* **Major**: Breaking API changes
* **Minor**: New features, backward compatible
* **Patch**: Bug fixes, performance improvements

Release Steps
~~~~~~~~~~~~~

1. Update version in `setup.py` and `package.json`
2. Update CHANGELOG.md
3. Create release branch
4. Run full test suite
5. Build and tag release
6. Deploy to staging
7. Deploy to production
8. Update documentation