const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs').promises;
const { spawn } = require('child_process');
const { Anthropic } = require('@anthropic-ai/sdk');

// RASSDB command paths
const RASSDB_BIN = path.join(__dirname, '..', 'bin');
const RASSDB_SEARCH = path.join(RASSDB_BIN, 'rassdb-search');
const RASSDB_STATS = path.join(RASSDB_BIN, 'rassdb-stats');

const app = express();
const PORT = process.env.PORT || 3000;

// Initialize Anthropic client if API key is available
const anthropicApiKey = process.env.ANTHROPIC_API_KEY;
let anthropicClient = null;
if (anthropicApiKey) {
    anthropicClient = new Anthropic({
        apiKey: anthropicApiKey
    });
}

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname)));

// Conversation memory
let conversationHistory = [];
let promptHistory = [];
let sessions = {};

// Load history on startup
async function loadHistory() {
    try {
        const promptData = await fs.readFile('prompt_history.json', 'utf8');
        promptHistory = JSON.parse(promptData);
    } catch (err) {
        promptHistory = [];
    }
    
    try {
        const convData = await fs.readFile('conversation_history.json', 'utf8');
        conversationHistory = JSON.parse(convData);
    } catch (err) {
        conversationHistory = [];
    }
}

// Save history
async function saveHistory() {
    await fs.writeFile('prompt_history.json', JSON.stringify(promptHistory.slice(-100), null, 2));
    await fs.writeFile('conversation_history.json', JSON.stringify(conversationHistory.slice(-20), null, 2));
}

// Session management endpoints
app.post('/sessions', (req, res) => {
    const sessionId = Date.now().toString(36) + Math.random().toString(36).substr(2);
    sessions[sessionId] = {
        id: sessionId,
        created: new Date().toISOString(),
        metadata: req.body.metadata || {},
        messages: []
    };
    res.json({ id: sessionId, created: sessions[sessionId].created });
});

app.get('/sessions/:id', (req, res) => {
    const session = sessions[req.params.id];
    if (!session) {
        return res.status(404).json({ error: 'Session not found' });
    }
    res.json(session);
});

app.delete('/sessions/:id', (req, res) => {
    delete sessions[req.params.id];
    res.json({ message: 'Session deleted' });
});

app.get('/sessions/:id/history', (req, res) => {
    const session = sessions[req.params.id];
    if (!session) {
        return res.status(404).json({ error: 'Session not found' });
    }
    res.json(session.messages || []);
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        version: '0.1.0',
        uptime: process.uptime()
    });
});

// RAG search endpoint
app.post('/api/search', async (req, res) => {
    const { query, limit = 5 } = req.body;
    
    try {
        // Call RASSDB search
        const pythonProcess = spawn(RASSDB_SEARCH, [
            query,
            '--semantic',
            '--limit', limit.toString(),
            '--db', path.join(__dirname, '.rassdb', 'example-chat-bot-nomic-embed-text-v1.5.rassdb'),
            '--format', 'json'
        ]);
        
        let output = '';
        let error = '';
        
        pythonProcess.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        pythonProcess.stderr.on('data', (data) => {
            error += data.toString();
        });
        
        pythonProcess.on('close', (code) => {
            if (code !== 0) {
                res.status(500).json({ error: error || 'RAG search failed' });
            } else {
                try {
                    const results = JSON.parse(output);
                    res.json(results);
                } catch (e) {
                    res.status(500).json({ error: 'Failed to parse search results' });
                }
            }
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Query endpoint for frontend
app.post('/query', async (req, res) => {
    const { query, session_id, top_k = 5 } = req.body;
    
    try {
        // Call RASSDB search
        const pythonProcess = spawn(RASSDB_SEARCH, [
            query,
            '--semantic',
            '--limit', top_k.toString(),
            '--db', path.join(__dirname, '.rassdb', 'example-chat-bot-nomic-embed-text-v1.5.rassdb'),
            '--format', 'json'
        ]);
        
        let output = '';
        let error = '';
        
        pythonProcess.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        pythonProcess.stderr.on('data', (data) => {
            error += data.toString();
        });
        
        pythonProcess.on('close', (code) => {
            if (code !== 0) {
                res.status(500).json({ error: error || 'Query failed' });
            } else {
                try {
                    const results = JSON.parse(output);
                    
                    // Store in session if provided
                    if (session_id && sessions[session_id]) {
                        sessions[session_id].messages.push({
                            query: query,
                            results: results,
                            timestamp: new Date().toISOString()
                        });
                    }
                    
                    res.json({
                        results: results,
                        total: results.length,
                        query_time: 0.1,
                        session_id: session_id
                    });
                } catch (e) {
                    res.status(500).json({ error: 'Failed to parse search results' });
                }
            }
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Get available models from Ollama
app.get('/api/models', async (req, res) => {
    try {
        const response = await fetch('http://localhost:11434/api/tags');
        const data = await response.json();
        res.json(data);
    } catch (error) {
        res.json({ models: [] });
    }
});

// Generate response with RAG context
app.post('/api/generate', async (req, res) => {
    const { prompt, model = 'qwen2.5-coder:7b-instruct', useRAG = true, modelType = 'ollama' } = req.body;
    
    // Add to prompt history
    promptHistory.push(prompt);
    
    // Set up SSE immediately
    res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
    });
    
    let contextualPrompt = prompt;
    let ragContext = '';
    let ragResults = null;
    
    // Search for relevant code context if RAG is enabled
    if (useRAG) {
        try {
            const ragLimit = req.body.ragLimit || 5;
            const searchProcess = spawn(RASSDB_SEARCH, [
                '-s',  // Use semantic search
                '--format', 'json',
                '--limit', ragLimit.toString(),
                prompt
            ]);
            
            let searchOutput = '';
            
            await new Promise((resolve, reject) => {
                searchProcess.stdout.on('data', (data) => {
                    searchOutput += data.toString();
                });
                
                searchProcess.stderr.on('data', (data) => {
                    console.error('RAG search error:', data.toString());
                });
                
                searchProcess.on('close', (code) => {
                    if (code === 0) {
                        try {
                            const results = JSON.parse(searchOutput);
                            if (results.length > 0) {
                                ragContext = '\n\nRelevant code context:\n\n';
                                results.forEach((result, idx) => {
                                    ragContext += `### Context ${idx + 1} (${result.file_path}:${result.start_line}-${result.end_line})\n`;
                                    ragContext += `Type: ${result.chunk_type}, Language: ${result.language}\n`;
                                    ragContext += '```' + (result.language || '') + '\n';
                                    ragContext += result.content + '\n';
                                    ragContext += '```\n\n';
                                });
                                
                                contextualPrompt = ragContext + '\nUser question: ' + prompt;
                                ragResults = results;
                                
                                // Send RAG results immediately via SSE
                                res.write(`data: ${JSON.stringify({
                                    type: 'rag',
                                    query: prompt,
                                    results: results,
                                    formattedContext: ragContext
                                })}\n\n`);
                            }
                        } catch (e) {
                            console.error('Failed to parse RAG results:', e);
                        }
                    }
                    resolve();
                });
            });
        } catch (error) {
            console.error('RAG search error:', error);
        }
    }
    
    // Build conversation context
    let messages = [];
    
    // Include recent conversation history
    conversationHistory.slice(-6).forEach(entry => {
        messages.push({ role: 'user', content: entry.prompt });
        messages.push({ role: 'assistant', content: entry.response });
    });
    
    // Add current prompt
    messages.push({ role: 'user', content: contextualPrompt });
    
    try {
        let fullResponse = '';
        
        if (modelType === 'anthropic' && anthropicClient) {
            // Use Anthropic API
            try {
                const stream = await anthropicClient.messages.create({
                    model: model || 'claude-3-5-sonnet-20241022',
                    messages: messages.map(msg => ({
                        role: msg.role === 'assistant' ? 'assistant' : 'user',
                        content: msg.content
                    })),
                    max_tokens: 4096,
                    stream: true
                });
                
                for await (const chunk of stream) {
                    if (chunk.type === 'content_block_delta' && chunk.delta.text) {
                        fullResponse += chunk.delta.text;
                        res.write(`data: ${JSON.stringify({
                            response: chunk.delta.text,
                            done: false
                        })}\n\n`);
                    }
                }
                
                // Save to history
                conversationHistory.push({
                    prompt: prompt,
                    response: fullResponse,
                    ragContext: ragContext,
                    timestamp: new Date().toISOString()
                });
                saveHistory();
                
                res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
                res.end();
                
            } catch (error) {
                console.error('Anthropic API error:', error);
                res.write(`data: ${JSON.stringify({ error: `Anthropic API error: ${error.message}` })}\n\n`);
                res.end();
            }
            
        } else {
            // Use Ollama API (default)
            const response = await fetch('http://localhost:11434/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: model,
                    messages: messages,
                    stream: true
                })
            });
            
            if (!response.ok) {
                throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
            }
            
            // Handle streaming response from fetch
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            try {
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value);
                    const lines = chunk.toString().split('\n').filter(line => line.trim());
                    
                    for (const line of lines) {
                        try {
                            const json = JSON.parse(line);
                            if (json.message && json.message.content) {
                                fullResponse += json.message.content;
                                res.write(`data: ${JSON.stringify({
                                    response: json.message.content,
                                    done: json.done
                                })}\n\n`);
                            }
                            
                            if (json.done) {
                                // Save to history
                                conversationHistory.push({
                                    prompt: prompt,
                                    response: fullResponse,
                                    ragContext: ragContext,
                                    timestamp: new Date().toISOString()
                                });
                                saveHistory();
                            }
                        } catch (e) {
                            // Ignore parse errors
                        }
                    }
                }
                
                res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
                res.end();
                
            } catch (error) {
                console.error('Stream reading error:', error);
                res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
                res.end();
            }
        }
        
    } catch (error) {
        console.error('Generate error:', error);
        // Only send JSON response if headers haven't been sent
        if (!res.headersSent) {
            res.status(500).json({ error: error.message });
        } else {
            // If streaming already started, send error through SSE
            res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
            res.end();
        }
    }
});

// Get conversation history
app.get('/api/history', (req, res) => {
    res.json(promptHistory.slice(-100));
});

// Clear history
app.delete('/api/history', async (req, res) => {
    promptHistory = [];
    conversationHistory = [];
    await saveHistory();
    res.json({ message: 'History cleared' });
});

// Database statistics
app.get('/api/stats', async (req, res) => {
    try {
        const statsProcess = spawn(RASSDB_STATS, [
            '--db', path.join(__dirname, '.rassdb', 'example-chat-bot-nomic-embed-text-v1.5.rassdb'),
            '--format', 'json'
        ]);
        
        let output = '';
        
        statsProcess.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        statsProcess.on('close', (code) => {
            if (code === 0) {
                try {
                    const stats = JSON.parse(output);
                    res.json(stats);
                } catch (e) {
                    res.json({});
                }
            } else {
                res.json({});
            }
        });
    } catch (error) {
        res.json({});
    }
});

// Get available models from Ollama
app.get('/api/models', async (req, res) => {
    try {
        const response = await fetch('http://localhost:11434/api/tags');
        if (!response.ok) {
            throw new Error('Ollama not responding');
        }
        const data = await response.json();
        res.json(data);
    } catch (error) {
        console.error('Failed to fetch models:', error);
        res.status(503).json({ 
            error: 'Ollama service not available',
            models: []
        });
    }
});

// Check API status for both Ollama and Anthropic
app.get('/api/status', async (req, res) => {
    const status = {
        ollama: { available: false },
        anthropic: { available: false }
    };
    
    // Check Ollama
    try {
        const response = await fetch('http://localhost:11434/api/tags');
        if (response.ok) {
            const data = await response.json();
            status.ollama = {
                available: true,
                models: data.models || []
            };
        }
    } catch (error) {
        status.ollama.error = error.message;
    }
    
    // Check Anthropic
    if (anthropicClient) {
        status.anthropic = {
            available: true,
            models: ['claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022']
        };
    } else {
        status.anthropic.error = 'API key not configured';
    }
    
    res.json(status);
});

// Start server
loadHistory().then(() => {
    app.listen(PORT, () => {
        console.log(`RASSDB Chat Example running at http://localhost:${PORT}`);
        console.log('Make sure:');
        console.log('  1. Ollama is running (ollama serve)');
        console.log('  2. Qwen2.5-Coder model is installed (ollama pull qwen2.5-coder:7b-instruct)');
        console.log('  3. RASSDB is installed (pip install -e ..)');
        console.log('  4. A database exists at .rassdb/example-chat-bot-nomic-embed-text-v1.5.rassdb');
        
        if (anthropicClient) {
            console.log('\n✓ Anthropic API key detected - Claude models available');
        } else {
            console.log('\n  For Anthropic Claude access: set ANTHROPIC_API_KEY environment variable');
        }
    });
});