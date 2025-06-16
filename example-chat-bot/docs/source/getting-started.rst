Getting Started
===============

This guide will help you set up and run the RASSDB Chat Bot on your local machine.

Prerequisites
-------------

Before you begin, ensure you have the following installed:

* Python 3.8 or higher
* Node.js 14.0 or higher
* npm (comes with Node.js)
* Git

Installation
------------

1. Clone the Repository
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/example/rassdb-chat.git
   cd rassdb-chat

2. Set Up Python Backend
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create a virtual environment
   python -m venv venv
   
   # Activate the virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   
   # Install the package in development mode
   pip install -e .
   
   # Or install just the requirements
   pip install -r requirements.txt

3. Set Up JavaScript Frontend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd src/frontend
   npm install
   
   # Copy environment variables
   cp .env.example .env

4. Configure Environment
~~~~~~~~~~~~~~~~~~~~~~~~

Edit ``src/frontend/.env`` to set your configuration:

.. code-block:: bash

   PORT=3000
   BACKEND_URL=http://localhost:8000
   SESSION_SECRET=your-secret-key-here

Running the Application
-----------------------

You'll need to run both the backend and frontend servers.

Start the Backend Server
~~~~~~~~~~~~~~~~~~~~~~~~

In one terminal:

.. code-block:: bash

   # From the project root
   rassdb-chat-server
   
   # Or run directly with Python
   python -m rassdb_chat.api.server

The backend will start on http://localhost:8000

Start the Frontend Server
~~~~~~~~~~~~~~~~~~~~~~~~~

In another terminal:

.. code-block:: bash

   cd src/frontend
   npm start

The frontend will start on http://localhost:3000

Verify Installation
-------------------

1. Open your browser to http://localhost:3000
2. You should see the RASSDB Chat Bot interface
3. Check that both server status indicators show "Online"
4. Try asking a simple query like "What is this project about?"

First Steps
-----------

Once the application is running:

1. **Create a Session**: The app automatically creates a session when you load the page
2. **Ask Questions**: Type natural language queries about your codebase
3. **View Results**: The bot will return relevant code snippets and explanations
4. **Manage Sessions**: Use the "Clear Session" button to start fresh

Example Queries
~~~~~~~~~~~~~~~

Try these example queries to get started:

* "Show me the main server code"
* "How does the MCP handler work?"
* "Find functions that handle user queries"
* "What API endpoints are available?"

Next Steps
----------

* Read the :doc:`architecture` guide to understand the system design
* Check the :doc:`api-reference` for detailed endpoint documentation
* Learn about the :doc:`mcp-protocol` for extending functionality
* See :doc:`deployment` for production setup instructions

Troubleshooting
---------------

If you encounter issues:

Backend Issues
~~~~~~~~~~~~~~

* Ensure Python 3.8+ is installed: ``python --version``
* Check that the virtual environment is activated
* Verify all dependencies are installed: ``pip list``
* Check backend logs for error messages

Frontend Issues
~~~~~~~~~~~~~~~

* Ensure Node.js 14+ is installed: ``node --version``
* Verify npm packages are installed: ``npm list``
* Check that the backend URL in ``.env`` is correct
* Look at browser console for JavaScript errors

Connection Issues
~~~~~~~~~~~~~~~~~

* Verify both servers are running
* Check that ports 3000 and 8000 are not in use
* Ensure firewall isn't blocking local connections
* Try accessing backend directly at http://localhost:8000/health