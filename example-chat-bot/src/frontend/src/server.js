/**
 * Express server for RASSDB Chat frontend
 * Communicates with Python backend via HTTP/JSON
 */

const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');
const cors = require('cors');
const session = require('express-session');
const morgan = require('morgan');
const path = require('path');
const { v4: uuidv4 } = require('uuid');

// Load environment variables
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

// Middleware
app.use(morgan('dev'));
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, '../public')));

// Session management
app.use(session({
    secret: process.env.SESSION_SECRET || 'rassdb-chat-secret',
    resave: false,
    saveUninitialized: true,
    cookie: { secure: false } // Set to true in production with HTTPS
}));

// Health check endpoint
app.get('/api/health', async (req, res) => {
    try {
        // Check backend health
        const backendHealth = await axios.get(`${BACKEND_URL}/health`);
        res.json({
            status: 'healthy',
            frontend: {
                version: '0.1.0',
                uptime: process.uptime()
            },
            backend: backendHealth.data
        });
    } catch (error) {
        res.status(503).json({
            status: 'unhealthy',
            error: 'Backend connection failed',
            details: error.message
        });
    }
});

// Create or get session
app.post('/api/session', async (req, res) => {
    try {
        if (!req.session.chatSessionId) {
            // Create new session in backend
            const response = await axios.post(`${BACKEND_URL}/sessions`, {
                metadata: {
                    frontend_session_id: req.sessionID,
                    user_agent: req.headers['user-agent']
                }
            });
            req.session.chatSessionId = response.data.id;
        }
        
        res.json({
            session_id: req.session.chatSessionId,
            frontend_session_id: req.sessionID
        });
    } catch (error) {
        console.error('Session creation failed:', error);
        res.status(500).json({
            error: 'Failed to create session',
            details: error.message
        });
    }
});

// Query endpoint - forwards to Python backend
app.post('/api/query', async (req, res) => {
    try {
        const { query, top_k = 5, filters } = req.body;
        
        if (!query) {
            return res.status(400).json({
                error: 'Query is required'
            });
        }
        
        // Get or create session
        if (!req.session.chatSessionId) {
            const sessionResponse = await axios.post(`${BACKEND_URL}/sessions`);
            req.session.chatSessionId = sessionResponse.data.id;
        }
        
        // Forward query to backend
        const response = await axios.post(`${BACKEND_URL}/query`, {
            query,
            session_id: req.session.chatSessionId,
            top_k,
            filters
        });
        
        res.json({
            results: response.data.results,
            total: response.data.total,
            query_time: response.data.query_time,
            session_id: response.data.session_id
        });
    } catch (error) {
        console.error('Query failed:', error);
        res.status(500).json({
            error: 'Query processing failed',
            details: error.message
        });
    }
});

// Get chat history
app.get('/api/history', async (req, res) => {
    try {
        if (!req.session.chatSessionId) {
            return res.json({ messages: [] });
        }
        
        const response = await axios.get(
            `${BACKEND_URL}/sessions/${req.session.chatSessionId}/history`
        );
        
        res.json({
            messages: response.data,
            session_id: req.session.chatSessionId
        });
    } catch (error) {
        console.error('History fetch failed:', error);
        res.status(500).json({
            error: 'Failed to fetch history',
            details: error.message
        });
    }
});

// Clear session
app.delete('/api/session', async (req, res) => {
    try {
        if (req.session.chatSessionId) {
            await axios.delete(
                `${BACKEND_URL}/sessions/${req.session.chatSessionId}`
            );
            delete req.session.chatSessionId;
        }
        
        res.json({
            message: 'Session cleared successfully'
        });
    } catch (error) {
        console.error('Session clear failed:', error);
        res.status(500).json({
            error: 'Failed to clear session',
            details: error.message
        });
    }
});

// Find similar code
app.post('/api/similar-code', async (req, res) => {
    try {
        const { code_snippet, language = 'python', top_k = 5 } = req.body;
        
        if (!code_snippet) {
            return res.status(400).json({
                error: 'Code snippet is required'
            });
        }
        
        const response = await axios.post(`${BACKEND_URL}/similar-code`, null, {
            params: { code_snippet, language, top_k }
        });
        
        res.json(response.data);
    } catch (error) {
        console.error('Similar code search failed:', error);
        res.status(500).json({
            error: 'Similar code search failed',
            details: error.message
        });
    }
});

// Index directory (admin endpoint)
app.post('/api/admin/index', async (req, res) => {
    try {
        const { directory, ignore_patterns } = req.body;
        
        if (!directory) {
            return res.status(400).json({
                error: 'Directory is required'
            });
        }
        
        const response = await axios.post(`${BACKEND_URL}/index`, {
            directory,
            ignore_patterns
        });
        
        res.json(response.data);
    } catch (error) {
        console.error('Indexing failed:', error);
        res.status(500).json({
            error: 'Indexing failed',
            details: error.message
        });
    }
});

// Serve index.html for all other routes (SPA support)
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, '../public/index.html'));
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Server error:', err);
    res.status(500).json({
        error: 'Internal server error',
        details: err.message
    });
});

// Start server
app.listen(PORT, () => {
    console.log(`RASSDB Chat Frontend Server running on http://localhost:${PORT}`);
    console.log(`Backend URL: ${BACKEND_URL}`);
});