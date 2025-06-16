/**
 * RASSDB Chat Bot Frontend Application
 */

class ChatApp {
    constructor() {
        this.sessionId = null;
        this.isLoading = false;
        this.initializeElements();
        this.attachEventListeners();
        this.initializeSession();
        this.checkServerStatus();
        
        // Check server status periodically
        setInterval(() => this.checkServerStatus(), 30000);
    }
    
    initializeElements() {
        this.elements = {
            chatHistory: document.getElementById('chat-history'),
            queryForm: document.getElementById('query-form'),
            queryInput: document.getElementById('query-input'),
            topK: document.getElementById('top-k'),
            sessionId: document.getElementById('session-id'),
            clearSession: document.getElementById('clear-session'),
            frontendStatus: document.getElementById('frontend-status'),
            backendStatus: document.getElementById('backend-status')
        };
    }
    
    attachEventListeners() {
        this.elements.queryForm.addEventListener('submit', (e) => this.handleSubmit(e));
        this.elements.clearSession.addEventListener('click', () => this.clearSession());
    }
    
    async initializeSession() {
        try {
            const response = await fetch('/api/session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                this.sessionId = data.session_id;
                this.elements.sessionId.textContent = this.sessionId;
                await this.loadHistory();
            }
        } catch (error) {
            console.error('Failed to initialize session:', error);
            this.showError('Failed to initialize session');
        }
    }
    
    async checkServerStatus() {
        try {
            const response = await fetch('/api/health');
            const data = await response.json();
            
            if (data.status === 'healthy') {
                this.elements.frontendStatus.textContent = 'Online';
                this.elements.frontendStatus.className = 'status-indicator healthy';
                
                if (data.backend && data.backend.status === 'healthy') {
                    this.elements.backendStatus.textContent = 'Online';
                    this.elements.backendStatus.className = 'status-indicator healthy';
                } else {
                    this.elements.backendStatus.textContent = 'Offline';
                    this.elements.backendStatus.className = 'status-indicator unhealthy';
                }
            }
        } catch (error) {
            this.elements.frontendStatus.textContent = 'Error';
            this.elements.frontendStatus.className = 'status-indicator unhealthy';
            this.elements.backendStatus.textContent = 'Unknown';
            this.elements.backendStatus.className = 'status-indicator unhealthy';
        }
    }
    
    async handleSubmit(event) {
        event.preventDefault();
        
        if (this.isLoading) return;
        
        const query = this.elements.queryInput.value.trim();
        if (!query) return;
        
        this.isLoading = true;
        this.elements.queryInput.disabled = true;
        
        // Add user message to chat
        this.addMessage('user', query);
        
        // Clear input
        this.elements.queryInput.value = '';
        
        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    top_k: parseInt(this.elements.topK.value)
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                this.handleQueryResponse(data);
            } else {
                const error = await response.json();
                this.showError(error.error || 'Query failed');
            }
        } catch (error) {
            console.error('Query error:', error);
            this.showError('Failed to send query');
        } finally {
            this.isLoading = false;
            this.elements.queryInput.disabled = false;
            this.elements.queryInput.focus();
        }
    }
    
    handleQueryResponse(data) {
        // Format results into a readable message
        let content = '';
        
        if (data.results && data.results.length > 0) {
            content = `Found ${data.total} result(s) in ${data.query_time.toFixed(2)}s:\n\n`;
            
            data.results.forEach((result, index) => {
                const metadata = result.metadata || {};
                content += `${index + 1}. **${metadata.file_path || 'Unknown file'}**`;
                if (metadata.line_number) {
                    content += ` (line ${metadata.line_number})`;
                }
                content += `\n   Score: ${(result.score || 0).toFixed(2)}\n`;
                content += `   ${result.content || 'No content available'}\n\n`;
            });
        } else {
            content = "I couldn't find any relevant information for your query.";
        }
        
        this.addMessage('assistant', content);
    }
    
    addMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const roleDiv = document.createElement('div');
        roleDiv.className = 'role';
        roleDiv.textContent = role === 'user' ? 'You' : 'Assistant';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'content';
        contentDiv.textContent = content;
        
        const timestampDiv = document.createElement('div');
        timestampDiv.className = 'timestamp';
        timestampDiv.textContent = new Date().toLocaleTimeString();
        
        messageDiv.appendChild(roleDiv);
        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(timestampDiv);
        
        this.elements.chatHistory.appendChild(messageDiv);
        this.elements.chatHistory.scrollTop = this.elements.chatHistory.scrollHeight;
    }
    
    showError(message) {
        this.addMessage('assistant', `Error: ${message}`);
    }
    
    async loadHistory() {
        try {
            const response = await fetch('/api/history');
            if (response.ok) {
                const data = await response.json();
                
                // Clear existing messages
                this.elements.chatHistory.innerHTML = '';
                
                // Add historical messages
                data.messages.forEach(message => {
                    this.addMessage(message.role, message.content);
                });
            }
        } catch (error) {
            console.error('Failed to load history:', error);
        }
    }
    
    async clearSession() {
        if (!confirm('Are you sure you want to clear the session?')) {
            return;
        }
        
        try {
            const response = await fetch('/api/session', {
                method: 'DELETE'
            });
            
            if (response.ok) {
                // Clear chat history
                this.elements.chatHistory.innerHTML = '';
                
                // Initialize new session
                await this.initializeSession();
                
                this.addMessage('assistant', 'Session cleared. Starting a new session.');
            }
        } catch (error) {
            console.error('Failed to clear session:', error);
            this.showError('Failed to clear session');
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new ChatApp();
});