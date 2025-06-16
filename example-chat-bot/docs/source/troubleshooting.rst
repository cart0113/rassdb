Troubleshooting
===============

This guide helps diagnose and resolve common issues with the RASSDB Chat Bot.

Common Issues
-------------

Backend Server Issues
~~~~~~~~~~~~~~~~~~~~~

**Problem: Backend server won't start**

Symptoms:
- Error: "Address already in use"
- Server crashes immediately
- Import errors

Solutions:

.. code-block:: bash

   # Check if port 8000 is in use
   lsof -i :8000  # macOS/Linux
   netstat -ano | findstr :8000  # Windows
   
   # Kill the process using the port
   kill -9 <PID>  # Use the PID from above
   
   # Or change the port
   uvicorn rassdb_chat.api.server:app --port 8001

**Problem: RASSDB connection fails**

Symptoms:
- "Failed to initialize RASSDB"
- Database file not found
- Permission denied errors

Solutions:

.. code-block:: bash

   # Check database file exists
   ls -la .rassdb/
   
   # Create directory if missing
   mkdir -p .rassdb
   
   # Fix permissions
   chmod 755 .rassdb
   chmod 644 .rassdb/*.rassdb
   
   # Verify environment variable
   echo $RASSDB_PATH

Frontend Server Issues
~~~~~~~~~~~~~~~~~~~~~~

**Problem: Cannot connect to backend**

Symptoms:
- "Backend connection failed" in UI
- CORS errors in browser console
- 404 errors on API calls

Solutions:

.. code-block:: javascript

   // Check backend URL in .env
   BACKEND_URL=http://localhost:8000  // Ensure this is correct
   
   // Verify CORS settings in backend
   // src/rassdb_chat/api/server.py
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["http://localhost:3000"],  // Add your frontend URL
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )

**Problem: Session management issues**

Symptoms:
- Sessions not persisting
- "Session not found" errors
- Random logouts

Solutions:

.. code-block:: bash

   # Check session secret is set
   echo $SESSION_SECRET
   
   # Generate a strong secret
   openssl rand -base64 32
   
   # Update .env file
   SESSION_SECRET=your-generated-secret

Query Processing Issues
~~~~~~~~~~~~~~~~~~~~~~~

**Problem: Queries return no results**

Symptoms:
- Empty results for all queries
- "No relevant information found"
- Timeout errors

Diagnostic steps:

1. **Check RASSDB index**:

   .. code-block:: bash

      # Verify index exists
      ls -la .rassdb/*.rassdb
      
      # Check index size (should be > 0)
      du -h .rassdb/*.rassdb
      
      # Re-index if necessary
      rassdb index --path ./src --output .rassdb/

2. **Test backend directly**:

   .. code-block:: bash

      # Test health endpoint
      curl http://localhost:8000/health
      
      # Test query endpoint
      curl -X POST http://localhost:8000/query \
           -H "Content-Type: application/json" \
           -d '{"query": "test query", "top_k": 5}'

3. **Check logs**:

   .. code-block:: python

      # Enable debug logging
      import logging
      logging.basicConfig(level=logging.DEBUG)

Performance Issues
~~~~~~~~~~~~~~~~~~

**Problem: Slow query responses**

Symptoms:
- Queries take > 5 seconds
- UI freezes during queries
- High CPU/memory usage

Solutions:

1. **Optimize RASSDB**:

   .. code-block:: python

      # Reduce result count
      top_k = 3  # Instead of 10
      
      # Add query filters
      filters = {"file_type": "python", "max_size": 10000}

2. **Enable caching**:

   .. code-block:: python

      # Add simple caching
      from functools import lru_cache
      
      @lru_cache(maxsize=100)
      async def cached_query(query_text: str, top_k: int):
          return await mcp_handler.query(query_text, top_k)

3. **Check system resources**:

   .. code-block:: bash

      # Monitor CPU and memory
      top  # Linux/macOS
      
      # Check disk I/O
      iotop  # Linux
      
      # Increase process limits if needed
      ulimit -n 4096  # Increase file descriptors

Installation Issues
-------------------

Python Dependencies
~~~~~~~~~~~~~~~~~~~

**Problem: Package installation fails**

.. code-block:: bash

   # Clear pip cache
   pip cache purge
   
   # Upgrade pip
   python -m pip install --upgrade pip
   
   # Install with verbose output
   pip install -v -r requirements.txt
   
   # Use different index if needed
   pip install -r requirements.txt -i https://pypi.org/simple

**Problem: Version conflicts**

.. code-block:: bash

   # Create fresh virtual environment
   python -m venv venv_fresh
   source venv_fresh/bin/activate
   
   # Install exact versions
   pip install -r requirements.txt --force-reinstall

Node.js Dependencies
~~~~~~~~~~~~~~~~~~~~

**Problem: npm install fails**

.. code-block:: bash

   # Clear npm cache
   npm cache clean --force
   
   # Delete node_modules
   rm -rf node_modules package-lock.json
   
   # Reinstall
   npm install
   
   # Use different registry if needed
   npm install --registry https://registry.npmjs.org/

Docker Issues
-------------

**Problem: Container build fails**

.. code-block:: bash

   # Clear Docker cache
   docker system prune -a
   
   # Build with no cache
   docker build --no-cache -t rassdb-chat-backend .
   
   # Check Docker logs
   docker logs <container-id>

**Problem: Containers can't communicate**

.. code-block:: yaml

   # Ensure services are on same network
   services:
     backend:
       networks:
         - rassdb-net
     frontend:
       networks:
         - rassdb-net
   
   networks:
     rassdb-net:
       driver: bridge

Database Issues
---------------

**Problem: Index corruption**

Symptoms:
- Unexpected errors during queries
- Inconsistent results
- Database file errors

Solutions:

.. code-block:: bash

   # Backup existing index
   cp .rassdb/main.rassdb .rassdb/main.rassdb.backup
   
   # Re-index from scratch
   rm .rassdb/main.rassdb
   rassdb index --path ./src --output .rassdb/main.rassdb
   
   # Verify index integrity
   rassdb verify --index .rassdb/main.rassdb

Debugging Techniques
--------------------

Enable Verbose Logging
~~~~~~~~~~~~~~~~~~~~~~

Backend logging:

.. code-block:: python

   # Set in environment
   export LOG_LEVEL=DEBUG
   
   # Or in code
   import logging
   logging.getLogger("rassdb_chat").setLevel(logging.DEBUG)

Frontend logging:

.. code-block:: javascript

   // Enable debug mode
   const DEBUG = true;
   
   if (DEBUG) {
       console.log('Query request:', request);
       console.log('Query response:', response);
   }

Network Debugging
~~~~~~~~~~~~~~~~~

1. **Browser DevTools**:
   - Open Network tab
   - Check request/response headers
   - Look for CORS errors
   - Verify request payloads

2. **Use curl for testing**:

   .. code-block:: bash

      # Test with verbose output
      curl -v -X POST http://localhost:8000/query \
           -H "Content-Type: application/json" \
           -d '{"query": "test"}'

3. **Check with Postman**:
   - Import API collection
   - Test individual endpoints
   - Save working examples

Memory Profiling
~~~~~~~~~~~~~~~~

Python memory debugging:

.. code-block:: python

   # Use memory_profiler
   from memory_profiler import profile
   
   @profile
   def memory_intensive_function():
       # Your code here
       pass

JavaScript memory debugging:

.. code-block:: javascript

   // Take heap snapshots in Chrome DevTools
   // Memory tab -> Take snapshot
   
   // Or use programmatic approach
   if (performance.memory) {
       console.log('Memory usage:', {
           used: performance.memory.usedJSHeapSize,
           total: performance.memory.totalJSHeapSize,
           limit: performance.memory.jsHeapSizeLimit
       });
   }

Error Messages Reference
------------------------

Backend Errors
~~~~~~~~~~~~~~

**"RASSDB not initialized"**
- Cause: Database connection not established
- Fix: Check database path and permissions

**"Session not found"**
- Cause: Invalid or expired session ID
- Fix: Create new session or check session storage

**"Query timeout"**
- Cause: Query took too long to process
- Fix: Simplify query or increase timeout

Frontend Errors
~~~~~~~~~~~~~~~

**"Network Error"**
- Cause: Backend unreachable
- Fix: Verify backend is running and accessible

**"Invalid response format"**
- Cause: Backend returned unexpected data
- Fix: Check API version compatibility

**"Session expired"**
- Cause: Session timeout reached
- Fix: Refresh page to create new session

Getting Help
------------

If you're still experiencing issues:

1. **Check the logs**: Both frontend and backend logs
2. **Search existing issues**: GitHub issues page
3. **Ask for help**:
   - Discord: [Project Discord]
   - GitHub Discussions
   - Stack Overflow with tag `rassdb-chat`

When reporting issues, include:
- Error messages (full stack trace)
- Steps to reproduce
- Environment details (OS, Python/Node versions)
- Relevant configuration files
- Log outputs

Quick Fixes Checklist
---------------------

Before deep debugging, try these quick fixes:

- [ ] Restart both servers
- [ ] Clear browser cache and cookies
- [ ] Check all required ports are free
- [ ] Verify environment variables are set
- [ ] Ensure database files exist
- [ ] Update to latest versions
- [ ] Try incognito/private browsing mode
- [ ] Disable browser extensions
- [ ] Check firewall settings
- [ ] Verify file permissions