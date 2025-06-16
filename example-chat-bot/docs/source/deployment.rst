Deployment
==========

This guide covers deploying the RASSDB Chat Bot to production environments.

Deployment Options
------------------

The application supports multiple deployment strategies:

1. **Docker Containers**: Recommended for most deployments
2. **Kubernetes**: For orchestrated, scalable deployments
3. **Cloud Platforms**: AWS, Google Cloud, Azure
4. **Traditional Servers**: Direct installation on Linux servers

Docker Deployment
-----------------

Create Docker images for both frontend and backend:

Backend Dockerfile
~~~~~~~~~~~~~~~~~~

.. code-block:: dockerfile

   # Dockerfile.backend
   FROM python:3.11-slim
   
   WORKDIR /app
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       gcc \
       && rm -rf /var/lib/apt/lists/*
   
   # Copy requirements first for better caching
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   # Copy application code
   COPY src/rassdb_chat ./rassdb_chat
   COPY setup.py .
   
   # Install the package
   RUN pip install -e .
   
   # Expose port
   EXPOSE 8000
   
   # Run the server
   CMD ["uvicorn", "rassdb_chat.api.server:app", \
        "--host", "0.0.0.0", "--port", "8000"]

Frontend Dockerfile
~~~~~~~~~~~~~~~~~~~

.. code-block:: dockerfile

   # Dockerfile.frontend
   FROM node:18-alpine
   
   WORKDIR /app
   
   # Copy package files
   COPY src/frontend/package*.json ./
   
   # Install dependencies
   RUN npm ci --only=production
   
   # Copy application code
   COPY src/frontend .
   
   # Expose port
   EXPOSE 3000
   
   # Run the server
   CMD ["node", "src/server.js"]

Docker Compose
~~~~~~~~~~~~~~

Use Docker Compose for local deployment:

.. code-block:: yaml

   # docker-compose.yml
   version: '3.8'
   
   services:
     backend:
       build:
         context: .
         dockerfile: Dockerfile.backend
       ports:
         - "8000:8000"
       environment:
         - RASSDB_PATH=/data/rassdb
       volumes:
         - rassdb_data:/data
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
         interval: 30s
         timeout: 10s
         retries: 3
   
     frontend:
       build:
         context: .
         dockerfile: Dockerfile.frontend
       ports:
         - "3000:3000"
       environment:
         - BACKEND_URL=http://backend:8000
         - SESSION_SECRET=${SESSION_SECRET}
       depends_on:
         - backend
       healthcheck:
         test: ["CMD", "wget", "--spider", "-q", "http://localhost:3000/api/health"]
         interval: 30s
         timeout: 10s
         retries: 3
   
   volumes:
     rassdb_data:

Kubernetes Deployment
---------------------

Deploy to Kubernetes for production scalability:

Backend Deployment
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # k8s/backend-deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: rassdb-chat-backend
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: rassdb-chat-backend
     template:
       metadata:
         labels:
           app: rassdb-chat-backend
       spec:
         containers:
         - name: backend
           image: your-registry/rassdb-chat-backend:latest
           ports:
           - containerPort: 8000
           env:
           - name: RASSDB_PATH
             value: /data/rassdb
           volumeMounts:
           - name: rassdb-storage
             mountPath: /data
           livenessProbe:
             httpGet:
               path: /health
               port: 8000
             initialDelaySeconds: 30
             periodSeconds: 10
           readinessProbe:
             httpGet:
               path: /health
               port: 8000
             initialDelaySeconds: 5
             periodSeconds: 5
         volumes:
         - name: rassdb-storage
           persistentVolumeClaim:
             claimName: rassdb-pvc

Service Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # k8s/services.yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: rassdb-chat-backend
   spec:
     selector:
       app: rassdb-chat-backend
     ports:
     - port: 8000
       targetPort: 8000
     type: ClusterIP
   ---
   apiVersion: v1
   kind: Service
   metadata:
     name: rassdb-chat-frontend
   spec:
     selector:
       app: rassdb-chat-frontend
     ports:
     - port: 80
       targetPort: 3000
     type: LoadBalancer

Production Configuration
------------------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Backend configuration:

.. code-block:: bash

   # Backend environment
   RASSDB_PATH=/var/lib/rassdb/data
   RASSDB_INDEX_PATH=/var/lib/rassdb/index
   LOG_LEVEL=INFO
   MAX_CONNECTIONS=100
   CACHE_TTL=3600

Frontend configuration:

.. code-block:: bash

   # Frontend environment
   NODE_ENV=production
   BACKEND_URL=https://api.rassdb-chat.example.com
   SESSION_SECRET=<strong-random-secret>
   SESSION_TIMEOUT=3600000
   RATE_LIMIT_WINDOW=60000
   RATE_LIMIT_MAX=100

Nginx Configuration
~~~~~~~~~~~~~~~~~~~

Use Nginx as a reverse proxy:

.. code-block:: nginx

   # /etc/nginx/sites-available/rassdb-chat
   server {
       listen 80;
       server_name rassdb-chat.example.com;
       return 301 https://$server_name$request_uri;
   }
   
   server {
       listen 443 ssl http2;
       server_name rassdb-chat.example.com;
       
       ssl_certificate /etc/ssl/certs/rassdb-chat.crt;
       ssl_certificate_key /etc/ssl/private/rassdb-chat.key;
       
       # Frontend
       location / {
           proxy_pass http://localhost:3000;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection 'upgrade';
           proxy_set_header Host $host;
           proxy_cache_bypass $http_upgrade;
       }
       
       # Backend API
       location /api/backend/ {
           rewrite ^/api/backend/(.*) /$1 break;
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }

Database Setup
--------------

RASSDB Configuration
~~~~~~~~~~~~~~~~~~~~

1. **Storage Requirements**:
   
   * Minimum 10GB for index storage
   * SSD recommended for performance
   * Regular backups of index files

2. **Indexing Strategy**:
   
   .. code-block:: bash
   
      # Initial indexing
      rassdb index --path /code/repository \
                  --output /var/lib/rassdb/data \
                  --ignore-file .rassdb/ignore-index
      
      # Incremental updates
      rassdb update --path /code/repository \
                   --index /var/lib/rassdb/data

3. **Performance Tuning**:
   
   * Adjust vector dimensions based on model
   * Configure similarity threshold
   * Set appropriate cache sizes

Monitoring and Logging
----------------------

Prometheus Metrics
~~~~~~~~~~~~~~~~~~

Export metrics for monitoring:

.. code-block:: python

   # In backend server
   from prometheus_client import Counter, Histogram, Gauge
   
   query_counter = Counter('rassdb_queries_total', 
                          'Total number of queries')
   query_duration = Histogram('rassdb_query_duration_seconds',
                            'Query duration in seconds')
   active_sessions = Gauge('rassdb_active_sessions',
                          'Number of active sessions')

Logging Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # logging.conf
   [loggers]
   keys=root,rassdb_chat
   
   [handlers]
   keys=console,file
   
   [formatters]
   keys=default
   
   [logger_root]
   level=INFO
   handlers=console,file
   
   [logger_rassdb_chat]
   level=INFO
   handlers=console,file
   qualname=rassdb_chat
   propagate=0
   
   [handler_console]
   class=StreamHandler
   level=INFO
   formatter=default
   args=(sys.stdout,)
   
   [handler_file]
   class=handlers.RotatingFileHandler
   level=INFO
   formatter=default
   args=('/var/log/rassdb-chat/app.log', 'a', 10485760, 5)
   
   [formatter_default]
   format=%(asctime)s - %(name)s - %(levelname)s - %(message)s

Security Hardening
------------------

SSL/TLS Configuration
~~~~~~~~~~~~~~~~~~~~~

1. Use strong cipher suites
2. Enable HSTS headers
3. Implement certificate pinning
4. Regular certificate rotation

Authentication
~~~~~~~~~~~~~~

For production, implement proper authentication:

.. code-block:: python

   # Example JWT authentication
   from fastapi import Depends, HTTPException, Security
   from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
   
   security = HTTPBearer()
   
   async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
       token = credentials.credentials
       # Verify JWT token
       if not is_valid_token(token):
           raise HTTPException(status_code=401, detail="Invalid token")
       return decode_token(token)

Rate Limiting
~~~~~~~~~~~~~

Implement rate limiting to prevent abuse:

.. code-block:: python

   from slowapi import Limiter
   from slowapi.util import get_remote_address
   
   limiter = Limiter(key_func=get_remote_address)
   
   @app.post("/query")
   @limiter.limit("30/minute")
   async def query(request: QueryRequest):
       # Process query
       pass

Backup and Recovery
-------------------

Backup Strategy
~~~~~~~~~~~~~~~

1. **Database Backups**:
   
   .. code-block:: bash
   
      # Daily backup script
      #!/bin/bash
      BACKUP_DIR="/backups/rassdb"
      DATE=$(date +%Y%m%d_%H%M%S)
      
      # Backup RASSDB data
      tar -czf "$BACKUP_DIR/rassdb_$DATE.tar.gz" /var/lib/rassdb/data
      
      # Keep only last 7 days
      find "$BACKUP_DIR" -name "rassdb_*.tar.gz" -mtime +7 -delete

2. **Session Data**:
   
   * Use Redis for session storage
   * Enable Redis persistence (AOF)
   * Regular Redis backups

Disaster Recovery
~~~~~~~~~~~~~~~~~

1. **Multi-region deployment** for high availability
2. **Database replication** for redundancy
3. **Automated failover** procedures
4. **Regular recovery testing**

Performance Optimization
------------------------

Caching Strategy
~~~~~~~~~~~~~~~~

1. **Query Cache**: Cache frequent queries
2. **Result Cache**: Store processed results
3. **Static Assets**: CDN for frontend assets
4. **Database Connection Pool**: Reuse connections

Load Testing
~~~~~~~~~~~~

.. code-block:: bash

   # Example load test with locust
   locust -f loadtest.py --host=https://rassdb-chat.example.com \
          --users=100 --spawn-rate=10

Scaling Guidelines
~~~~~~~~~~~~~~~~~~

* **Horizontal Scaling**: Add more backend instances
* **Vertical Scaling**: Increase CPU/memory for complex queries
* **Database Sharding**: Split large indices
* **Queue Management**: Use Celery for async tasks