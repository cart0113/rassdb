// RASSDB Chat Example Web UI
const API_BASE = 'http://localhost:3000';

// State
let promptHistory = [];
let currentHistoryIndex = -1;
let isGenerating = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeUI();
    checkModelStatus();
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
}

// Dark mode
function updateDarkMode(isDark) {
    document.documentElement.classList.toggle('light-mode', !isDark);
    document.querySelector('.dark-mode-label').style.display = isDark ? 'inline' : 'none';
    document.querySelector('.light-mode-label').style.display = isDark ? 'none' : 'inline';
}

// Check model status
async function checkModelStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/models`);
        const data = await response.json();
        
        const modelStatus = document.getElementById('modelStatus');
        if (data.models && data.models.length > 0) {
            // Find Qwen model
            const qwenModel = data.models.find(m => m.name.includes('qwen2.5-coder'));
            if (qwenModel) {
                modelStatus.innerHTML = '<span class="text-success">✓ Model connected</span>';
            } else {
                modelStatus.innerHTML = '<span class="text-warning">⚠ Qwen2.5-Coder not found</span>';
            }
            
            // Update model dropdown
            updateModelDropdown(data.models);
        } else {
            modelStatus.innerHTML = '<span class="text-danger">✗ No connection to Ollama</span>';
        }
    } catch (error) {
        document.getElementById('modelStatus').innerHTML = 
            '<span class="text-danger">✗ No connection to server</span>';
    }
}

// Update model dropdown
function updateModelDropdown(models) {
    const select = document.getElementById('modelSelect');
    const currentValue = select.value;
    
    select.innerHTML = '';
    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.name;
        option.textContent = model.name;
        select.appendChild(option);
    });
    
    // Restore selection if possible
    if (models.find(m => m.name === currentValue)) {
        select.value = currentValue;
    }
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
    
    // Show thinking indicator
    const thinkingDiv = addThinkingIndicator();
    
    try {
        const model = document.getElementById('modelSelect').value;
        const useRAG = document.getElementById('ragToggle').checked;
        
        const response = await fetch(`${API_BASE}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt, model, useRAG })
        });
        
        if (!response.ok) throw new Error('Generation failed');
        
        // Remove thinking indicator
        thinkingDiv.remove();
        
        // Create AI message container
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message ai';
        document.getElementById('chatMessages').appendChild(messageDiv);
        
        // Stream response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let responseText = '';
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        if (data.error) {
                            // Handle error in stream
                            messageDiv.innerHTML = `<span class="text-danger">Error: ${data.error}</span>`;
                            break;
                        } else if (data.response) {
                            responseText += data.response;
                            messageDiv.innerHTML = processMarkdown(responseText);
                            scrollToBottom();
                        }
                    } catch (e) {
                        // Ignore parse errors
                    }
                }
            }
        }
        
        // Apply syntax highlighting to any code blocks
        messageDiv.querySelectorAll('pre code').forEach(block => {
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
    const codeBlocks = container.querySelectorAll('pre code');
    
    codeBlocks.forEach((block, index) => {
        const pre = block.parentElement;
        
        // Skip if already processed
        if (pre.parentElement && pre.parentElement.classList.contains('code-container')) {
            return;
        }
        
        const language = block.className.match(/language-(\w+)/)?.[1] || 'plaintext';
        
        // Create wrapper
        const wrapper = document.createElement('div');
        wrapper.className = 'code-block-wrapper';
        
        // Create controls bar
        const controls = document.createElement('div');
        controls.className = 'code-controls';
        controls.innerHTML = `
            <span class="code-language">${language}</span>
            <button class="copy-btn" data-clipboard-target="#code-${Date.now()}-${index}">
                Copy
            </button>
        `;
        
        // Add ID to code block for clipboard
        block.id = `code-${Date.now()}-${index}`;
        
        // Add line numbers
        const lines = block.textContent.split('\n');
        const lineNumbers = document.createElement('div');
        lineNumbers.className = 'line-numbers';
        
        lines.forEach((_, i) => {
            const lineNum = document.createElement('div');
            lineNum.textContent = i + 1;
            lineNumbers.appendChild(lineNum);
        });
        
        // Create code container
        const codeContainer = document.createElement('div');
        codeContainer.className = 'code-container';
        
        // Insert wrapper before pre
        pre.parentElement.insertBefore(wrapper, pre);
        
        // Build the structure
        wrapper.appendChild(controls);
        wrapper.appendChild(codeContainer);
        codeContainer.appendChild(lineNumbers);
        codeContainer.appendChild(pre);
    });
    
    // Reinitialize clipboard for new buttons
    new ClipboardJS('.copy-btn').on('success', function(e) {
        const btn = e.trigger;
        const originalText = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => {
            btn.textContent = originalText;
        }, 2000);
    });
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

// Scroll to bottom
function scrollToBottom() {
    const messages = document.getElementById('chatMessages');
    messages.scrollTop = messages.scrollHeight;
}

// Periodic status check
setInterval(checkModelStatus, 30000);