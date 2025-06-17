// RASSDB Chat Example Web UI
const API_BASE = 'http://localhost:3000';

// State
let promptHistory = [];
let currentHistoryIndex = -1;
let isGenerating = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeUI();
    checkAPIStatus();
    loadPromptHistory();
    loadDBStats();
    
    // Set focus to input
    document.getElementById('promptInput').focus();
});

// UI Initialization
function initializeUI() {
    // Submit button
    document.getElementById('submitBtn').addEventListener('click', submitPrompt);
    
    // Enter key submission (Shift+Enter for new line)
    document.getElementById('promptInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submitPrompt();
        }
    });
    
    // History navigation
    document.getElementById('promptInput').addEventListener('keydown', (e) => {
        if (e.key === 'ArrowUp') {
            e.preventDefault();
            navigateHistory(-1);
        } else if (e.key === 'ArrowDown') {
            e.preventDefault();
            navigateHistory(1);
        }
    });
    
    // Clear buttons
    document.getElementById('clearHistoryBtn').addEventListener('click', clearHistory);
    document.getElementById('clearChatBtn').addEventListener('click', clearChat);
    
    // Dark mode toggle
    const darkModeToggle = document.getElementById('darkModeToggle');
    const savedDarkMode = localStorage.getItem('darkMode') !== 'false';
    darkModeToggle.checked = savedDarkMode;
    updateDarkMode(savedDarkMode);
    
    darkModeToggle.addEventListener('change', (e) => {
        updateDarkMode(e.target.checked);
        localStorage.setItem('darkMode', e.target.checked);
    });
    
    // Model selection
    const modelSelect = document.getElementById('modelSelect');
    const savedModel = localStorage.getItem('selectedModel');
    if (savedModel) {
        modelSelect.value = savedModel;
    }
    
    modelSelect.addEventListener('change', (e) => {
        localStorage.setItem('selectedModel', e.target.value);
    });
    
    // Initialize clipboard
    new ClipboardJS('.copy-btn');
    
    // RAG limit slider
    const ragLimitSlider = document.getElementById('ragLimitSlider');
    const ragLimitValue = document.getElementById('ragLimitValue');
    const savedLimit = localStorage.getItem('ragLimit') || '10';
    ragLimitSlider.value = savedLimit;
    ragLimitValue.textContent = savedLimit;
    
    ragLimitSlider.addEventListener('input', (e) => {
        ragLimitValue.textContent = e.target.value;
        localStorage.setItem('ragLimit', e.target.value);
    });
    
    // Model provider radio buttons
    const providerRadios = document.querySelectorAll('input[name="modelProvider"]');
    const savedProvider = localStorage.getItem('modelProvider') || 'ollama';
    document.querySelector(`input[name="modelProvider"][value="${savedProvider}"]`).checked = true;
    
    providerRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            localStorage.setItem('modelProvider', e.target.value);
            updateModelDropdown(e.target.value);
        });
    });
    
    // Initialize model dropdown based on saved provider
    updateModelDropdown(savedProvider);
}

// Dark mode
function updateDarkMode(isDark) {
    document.documentElement.classList.toggle('light-mode', !isDark);
    document.querySelector('.dark-mode-label').style.display = isDark ? 'inline' : 'none';
    document.querySelector('.light-mode-label').style.display = isDark ? 'none' : 'inline';
}

// Check API status
async function checkAPIStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        const data = await response.json();
        
        const modelStatus = document.getElementById('modelStatus');
        let statusHTML = '';
        
        if (data.ollama && data.ollama.available) {
            statusHTML += '<div class="text-success">✓ Ollama connected</div>';
        } else {
            statusHTML += '<div class="text-danger">✗ Ollama not available</div>';
        }
        
        if (data.anthropic && data.anthropic.available) {
            statusHTML += '<div class="text-success">✓ Anthropic API ready</div>';
        } else {
            statusHTML += '<div class="text-warning">⚠ Anthropic API not configured</div>';
        }
        
        modelStatus.innerHTML = statusHTML;
        
        // Store available models
        window.availableModels = {
            ollama: data.ollama.models || [],
            anthropic: data.anthropic.models || []
        };
        
        // Update dropdown based on current provider
        const currentProvider = document.querySelector('input[name="modelProvider"]:checked').value;
        updateModelDropdown(currentProvider);
        
    } catch (error) {
        document.getElementById('modelStatus').innerHTML = 
            '<span class="text-danger">✗ No connection to server</span>';
    }
}

// Update model dropdown
function updateModelDropdown(provider) {
    const select = document.getElementById('modelSelect');
    const currentValue = select.value;
    
    select.innerHTML = '';
    
    if (provider === 'anthropic') {
        // Anthropic models
        const anthropicModels = [
            { value: 'claude-3-5-sonnet-20241022', text: 'Claude 3.5 Sonnet' },
            { value: 'claude-3-5-haiku-20241022', text: 'Claude 3.5 Haiku' }
        ];
        
        anthropicModels.forEach(model => {
            const option = document.createElement('option');
            option.value = model.value;
            option.textContent = model.text;
            select.appendChild(option);
        });
        
        // Try to restore previous Anthropic selection
        const savedAnthropicModel = localStorage.getItem('selectedAnthropicModel');
        if (savedAnthropicModel && anthropicModels.find(m => m.value === savedAnthropicModel)) {
            select.value = savedAnthropicModel;
        }
        
    } else {
        // Ollama models
        const models = window.availableModels?.ollama || [];
        if (models.length === 0) {
            // Default Ollama models if not connected
            const defaultModels = [
                'qwen2.5-coder:7b-instruct',
                'qwen2.5-coder:3b-instruct'
            ];
            defaultModels.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                select.appendChild(option);
            });
        } else {
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.name;
                option.textContent = model.name;
                select.appendChild(option);
            });
        }
        
        // Try to restore previous Ollama selection
        const savedOllamaModel = localStorage.getItem('selectedOllamaModel');
        if (savedOllamaModel && (models.find(m => m.name === savedOllamaModel) || select.querySelector(`option[value="${savedOllamaModel}"]`))) {
            select.value = savedOllamaModel;
        }
    }
    
    // Save model selection per provider
    select.addEventListener('change', (e) => {
        if (provider === 'anthropic') {
            localStorage.setItem('selectedAnthropicModel', e.target.value);
        } else {
            localStorage.setItem('selectedOllamaModel', e.target.value);
        }
    });
}

// Load database statistics
async function loadDBStats() {
    try {
        const response = await fetch(`${API_BASE}/api/stats`);
        const stats = await response.json();
        
        if (stats.total_chunks) {
            const statsDiv = document.getElementById('dbStats');
            const content = document.getElementById('dbStatsContent');
            
            let html = `
                <div>Total chunks: ${stats.total_chunks}</div>
                <div>Files indexed: ${stats.unique_files}</div>
            `;
            
            if (stats.by_language && Object.keys(stats.by_language).length > 0) {
                html += '<div class="mt-2">Languages:</div>';
                for (const [lang, count] of Object.entries(stats.by_language)) {
                    html += `<div class="ms-2">${lang}: ${count}</div>`;
                }
            }
            
            content.innerHTML = html;
            statsDiv.style.display = 'block';
        }
    } catch (error) {
        console.error('Failed to load DB stats:', error);
    }
}

// Load prompt history
async function loadPromptHistory() {
    try {
        const response = await fetch(`${API_BASE}/api/history`);
        promptHistory = await response.json();
    } catch (error) {
        console.error('Failed to load history:', error);
    }
}

// Navigate history
function navigateHistory(direction) {
    if (promptHistory.length === 0) return;
    
    if (direction === -1) { // Up
        if (currentHistoryIndex < promptHistory.length - 1) {
            currentHistoryIndex++;
            document.getElementById('promptInput').value = 
                promptHistory[promptHistory.length - 1 - currentHistoryIndex];
        }
    } else { // Down
        if (currentHistoryIndex > 0) {
            currentHistoryIndex--;
            document.getElementById('promptInput').value = 
                promptHistory[promptHistory.length - 1 - currentHistoryIndex];
        } else if (currentHistoryIndex === 0) {
            currentHistoryIndex = -1;
            document.getElementById('promptInput').value = '';
        }
    }
}

// Submit prompt
async function submitPrompt() {
    if (isGenerating) return;
    
    const promptInput = document.getElementById('promptInput');
    const prompt = promptInput.value.trim();
    
    if (!prompt) return;
    
    isGenerating = true;
    document.getElementById('submitBtn').disabled = true;
    
    // Add user message
    addMessage(prompt, 'user');
    
    // Clear input
    promptInput.value = '';
    currentHistoryIndex = -1;
    
    // Clear previous RAG results
    document.getElementById('ragResults').innerHTML = '';
    
    // Show thinking indicator
    const thinkingDiv = addThinkingIndicator();
    
    // Show RAG thinking indicator if enabled
    const useRAG = document.getElementById('ragToggle').checked;
    let ragThinking = null;
    let ragStartTime = null;
    if (useRAG) {
        ragThinking = addRagThinkingIndicator();
        ragStartTime = Date.now();
    }
    
    try {
        const model = document.getElementById('modelSelect').value;
        const modelType = document.querySelector('input[name="modelProvider"]:checked').value;
        
        // Generate LLM response
        const ragLimit = parseInt(document.getElementById('ragLimitSlider').value);
        const response = await fetch(`${API_BASE}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt, model, modelType, useRAG, ragLimit })
        });
        
        if (!response.ok) throw new Error('Generation failed');
        
        // Create AI message container (but don't add it yet)
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message ai';
        let messageAdded = false;
        
        // Stream response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let responseText = '';
        let buffer = '';
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        if (data.type === 'rag') {
                            // Handle RAG results
                            if (ragThinking) {
                                ragThinking.remove();
                                ragThinking = null;
                            }
                            const ragEndTime = Date.now();
                            const ragDuration = ragStartTime ? (ragEndTime - ragStartTime) / 1000 : null;
                            displayRagQuery(data.query, ragDuration);
                            displayRagResults(data.results, data.formattedContext);
                        } else if (data.error) {
                            // Handle error in stream
                            messageDiv.innerHTML = `<span class="text-danger">Error: ${data.error}</span>`;
                            break;
                        } else if (data.response) {
                            // Remove thinking indicator and add message div on first LLM response
                            if (!messageAdded) {
                                thinkingDiv.remove();
                                document.getElementById('chatMessages').appendChild(messageDiv);
                                messageAdded = true;
                            }
                            responseText += data.response;
                            // Update content without rebuilding entire DOM
                            updateMessageContent(messageDiv, responseText);
                        }
                    } catch (e) {
                        // Ignore parse errors
                    }
                }
            }
        }
        
        // Process remaining buffer
        if (buffer && buffer.startsWith('data: ')) {
            try {
                const data = JSON.parse(buffer.slice(6));
                if (data.response) {
                    responseText += data.response;
                    updateMessageContent(messageDiv, responseText);
                }
            } catch (e) {
                // Ignore parse errors
            }
        }
        
        // Final syntax highlighting
        messageDiv.querySelectorAll('pre code:not(.hljs)').forEach(block => {
            hljs.highlightElement(block);
        });
        
        // Process code blocks for VS Code styling
        processCodeBlocks(messageDiv);
        
    } catch (error) {
        thinkingDiv.remove();
        addMessage('Error: ' + error.message, 'ai');
    } finally {
        isGenerating = false;
        document.getElementById('submitBtn').disabled = false;
        promptInput.focus();
        
        // Reload history
        loadPromptHistory();
    }
}

// Add message to chat
function addMessage(content, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    if (type === 'user') {
        // Escape HTML for user messages
        messageDiv.textContent = content;
    } else {
        // Process markdown for AI messages
        messageDiv.innerHTML = processMarkdown(content);
        
        // Apply syntax highlighting
        messageDiv.querySelectorAll('pre code').forEach(block => {
            hljs.highlightElement(block);
        });
        
        // Process code blocks for VS Code styling
        processCodeBlocks(messageDiv);
    }
    
    document.getElementById('chatMessages').appendChild(messageDiv);
    scrollToBottom();
}

// Add thinking indicator
function addThinkingIndicator() {
    const thinkingDiv = document.createElement('div');
    thinkingDiv.className = 'message ai thinking';
    thinkingDiv.innerHTML = `
        <div class="spinner-border spinner-border-sm text-primary me-2" role="status">
            <span class="visually-hidden">Loading...</span>
        </div>
        Thinking...
    `;
    document.getElementById('chatMessages').appendChild(thinkingDiv);
    scrollToBottom();
    return thinkingDiv;
}

// Process markdown
function processMarkdown(text) {
    return marked.parse(text, {
        gfm: true,
        breaks: true,
        highlight: function(code, lang) {
            if (lang && hljs.getLanguage(lang)) {
                return hljs.highlight(code, { language: lang }).value;
            }
            return hljs.highlightAuto(code).value;
        }
    });
}

// Process code blocks for VS Code styling
function processCodeBlocks(container) {
    // Don't process code blocks - let the default markdown rendering handle it
    // This was causing the line numbers to appear before the code
    return;
}

// Clear history
async function clearHistory() {
    if (confirm('Clear all prompt history?')) {
        try {
            await fetch(`${API_BASE}/api/history`, { method: 'DELETE' });
            promptHistory = [];
            currentHistoryIndex = -1;
        } catch (error) {
            console.error('Failed to clear history:', error);
        }
    }
}

// Clear chat
function clearChat() {
    document.getElementById('chatMessages').innerHTML = '';
}

// Display RAG query
function displayRagQuery(query, duration) {
    const ragContainer = document.getElementById('ragResults');
    const queryDiv = document.createElement('div');
    queryDiv.className = 'rag-query';
    
    let timingText = '';
    if (duration !== null && duration !== undefined) {
        timingText = `<div class="rag-timing" style="color: #4caf50; font-weight: bold; margin-bottom: 0.5rem;">RAG results retrieved in ${duration.toFixed(2)} seconds</div>`;
    }
    
    queryDiv.innerHTML = `
        ${timingText}
        <div class="rag-query-header">RAG Query:</div>
        <div class="rag-query-text">${escapeHtml(query)}</div>
    `;
    ragContainer.appendChild(queryDiv);
}

// Display RAG results
function displayRagResults(results, formattedContext) {
    const ragContainer = document.getElementById('ragResults');
    
    
    if (!results || results.length === 0) {
        const noResults = document.createElement('div');
        noResults.className = 'text-muted';
        noResults.textContent = 'No relevant code found.';
        ragContainer.appendChild(noResults);
        return;
    }
    
    const resultsHeader = document.createElement('div');
    resultsHeader.className = 'rag-results-header';
    resultsHeader.textContent = `Found ${results.length} relevant code chunks:`;
    ragContainer.appendChild(resultsHeader);
    
    results.forEach((result, index) => {
        const resultDiv = document.createElement('div');
        resultDiv.className = 'rag-result';
        
        // Build header with available fields
        let headerText = '';
        if (result.file_path) {
            headerText = result.file_path;
            if (result.start_line !== undefined) {
                headerText += `:${result.start_line}`;
                if (result.end_line !== undefined) {
                    headerText += `-${result.end_line}`;
                }
            }
        } else {
            headerText = `Result ${index + 1}`;
        }
        
        const header = document.createElement('div');
        header.className = 'rag-result-header';
        // Use similarity field from RASSDB results
        const score = result.similarity || result.score;
        const scoreText = score !== undefined && score !== null ? score.toFixed(3) : '—';
        header.innerHTML = `
            <span>${headerText}</span>
            <span class="rag-result-score">Score: ${scoreText}</span>
        `;
        
        // Display metadata if available
        if (result.chunk_type || result.language) {
            const metadata = document.createElement('div');
            metadata.className = 'rag-result-metadata';
            metadata.style.fontSize = '0.85rem';
            metadata.style.opacity = '0.8';
            metadata.style.marginBottom = '0.5rem';
            metadata.textContent = `Type: ${result.chunk_type || 'unknown'}, Language: ${result.language || 'unknown'}`;
            resultDiv.appendChild(metadata);
        }
        
        const content = document.createElement('pre');
        const codeText = result.content || result.text || JSON.stringify(result, null, 2);
        content.innerHTML = `<code class="language-${result.language || 'plaintext'}">${escapeHtml(codeText)}</code>`;
        
        resultDiv.appendChild(header);
        resultDiv.appendChild(content);
        ragContainer.appendChild(resultDiv);
        
        // Apply syntax highlighting
        hljs.highlightElement(content.querySelector('code'));
    });
    
    // Add formatted context section showing exactly what LLM receives
    if (formattedContext) {
        const contextDiv = document.createElement('div');
        contextDiv.className = 'rag-formatted-context';
        contextDiv.innerHTML = `
            <div class="rag-context-header">Exact context sent to LLM:</div>
            <pre class="rag-context-pre"><code>${escapeHtml(formattedContext)}</code></pre>
        `;
        ragContainer.appendChild(contextDiv);
    }
}

// Add RAG thinking indicator
function addRagThinkingIndicator() {
    const ragContainer = document.getElementById('ragResults');
    const thinkingDiv = document.createElement('div');
    thinkingDiv.className = 'rag-thinking';
    thinkingDiv.innerHTML = `
        <div class="spinner-border spinner-border-sm text-primary me-2" role="status">
            <span class="visually-hidden">Searching...</span>
        </div>
        Searching codebase...
    `;
    ragContainer.appendChild(thinkingDiv);
    return thinkingDiv;
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Scroll to bottom
function scrollToBottom() {
    const messages = document.getElementById('chatMessages');
    if (messages && messages.parentElement) {
        messages.parentElement.scrollTop = messages.parentElement.scrollHeight;
    }
}

// Update message content without rebuilding DOM
function updateMessageContent(messageDiv, content) {
    // Store scroll position
    const scrollPos = messageDiv.parentElement.scrollTop;
    const wasAtBottom = Math.abs(messageDiv.parentElement.scrollHeight - messageDiv.parentElement.scrollTop - messageDiv.parentElement.clientHeight) < 50;
    
    // Update content directly
    messageDiv.innerHTML = processMarkdown(content);
    
    // Apply syntax highlighting to new code blocks
    messageDiv.querySelectorAll('pre code:not(.hljs)').forEach(block => {
        hljs.highlightElement(block);
    });
    
    // Restore scroll position or scroll to bottom
    if (wasAtBottom) {
        scrollToBottom();
    } else {
        messageDiv.parentElement.scrollTop = scrollPos;
    }
}

// Periodic status check
setInterval(checkAPIStatus, 30000);