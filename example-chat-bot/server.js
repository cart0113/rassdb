const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs').promises;
const { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname)));

// Conversation memory
let conversationHistory = [];
let promptHistory = [];

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

// RAG search endpoint
app.post('/api/search', async (req, res) => {
    const { query, limit = 5 } = req.body;
    
    try {
        // Call RASSDB search
        const pythonProcess = spawn('rassdb-search', [
            query,
            '--semantic',
            '--limit', limit.toString(),
            '--db', path.join(__dirname, '..', 'code_rag.db'),
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
    const { prompt, model = 'qwen2.5-coder:7b-instruct', useRAG = true } = req.body;
    
    // Add to prompt history
    promptHistory.push(prompt);
    
    let contextualPrompt = prompt;
    let ragContext = '';
    
    // Search for relevant code context if RAG is enabled
    if (useRAG) {
        try {
            const searchProcess = spawn('rassdb-search', [
                prompt,
                '--semantic',
                '--limit', '5',
                '--db', path.join(__dirname, '..', 'code_rag.db'),
                '--format', 'json'
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
        
        // Set up SSE
        res.writeHead(200, {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        });
        
        let fullResponse = '';
        
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
        const statsProcess = spawn('rassdb-stats', [
            '--db', path.join(__dirname, '..', 'code_rag.db'),
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

// Start server
loadHistory().then(() => {
    app.listen(PORT, () => {
        console.log(`RASSDB Chat Example running at http://localhost:${PORT}`);
        console.log('Make sure:');
        console.log('  1. Ollama is running (ollama serve)');
        console.log('  2. Qwen2.5-Coder model is installed (ollama pull qwen2.5-coder:7b-instruct)');
        console.log('  3. RASSDB is installed (pip install -e ..)');
        console.log('  4. A database exists at ../code_rag.db');
    });
});