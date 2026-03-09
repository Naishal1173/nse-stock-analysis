// AI Chatbot for NSE Stock Analysis
let chatbotOpen = false;
let sessionId = 'session_' + Date.now();

// Initialize chatbot
document.addEventListener('DOMContentLoaded', function() {
    console.log('🤖 Initializing chatbot...');
    createChatbotUI();
    
    // Wait a bit for DOM to be fully ready
    setTimeout(() => {
        setupChatbotEvents();
        console.log('✅ Chatbot initialized successfully');
    }, 100);
});

function createChatbotUI() {
    const chatbotHTML = `
        <!-- Chatbot Button -->
        <div id="chatbotButton" class="chatbot-button" title="Ask AI Assistant" style="position: fixed; bottom: 24px; right: 24px; z-index: 10000;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
            </svg>
        </div>

        <!-- Chatbot Window -->
        <div id="chatbotWindow" class="chatbot-window" style="display: none; position: fixed; bottom: 24px; right: 24px; z-index: 10000;">
            <div class="chatbot-header">
                <div class="chatbot-header-content">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"></circle>
                        <path d="M8 14s1.5 2 4 2 4-2 4-2"></path>
                        <line x1="9" y1="9" x2="9.01" y2="9"></line>
                        <line x1="15" y1="9" x2="15.01" y2="9"></line>
                    </svg>
                    <div>
                        <div class="chatbot-title">AI Assistant</div>
                        <div class="chatbot-status">Online</div>
                    </div>
                </div>
                <div class="chatbot-header-actions">
                    <button id="clearChatBtn" class="chatbot-action-btn" title="Clear conversation">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="3 6 5 6 21 6"></polyline>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        </svg>
                    </button>
                    <button id="closeChatBtn" class="chatbot-action-btn" title="Close">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="18" y1="6" x2="6" y2="18"></line>
                            <line x1="6" y1="6" x2="18" y2="18"></line>
                        </svg>
                    </button>
                </div>
            </div>
            
            <div id="chatbotMessages" class="chatbot-messages">
                <div class="chatbot-message bot-message">
                    <div class="message-avatar">🤖</div>
                    <div class="message-content">
                        <div class="message-text">
                            Hi! I'm your AI assistant for NSE Stock Analysis. I can help you:
                            <br><br>
                            • Navigate to different pages<br>
                            • Explain technical indicators<br>
                            • Guide you through features<br>
                            • Answer trading questions<br>
                            <br>
                            What would you like to know?
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="chatbot-input-container">
                <input 
                    type="text" 
                    id="chatbotInput" 
                    class="chatbot-input" 
                    placeholder="Ask me anything..."
                    autocomplete="off"
                />
                <button id="sendChatBtn" class="chatbot-send-btn">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="22" y1="2" x2="11" y2="13"></line>
                        <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                    </svg>
                </button>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', chatbotHTML);
}

function setupChatbotEvents() {
    const chatbotButton = document.getElementById('chatbotButton');
    const chatbotWindow = document.getElementById('chatbotWindow');
    const closeChatBtn = document.getElementById('closeChatBtn');
    const clearChatBtn = document.getElementById('clearChatBtn');
    const sendChatBtn = document.getElementById('sendChatBtn');
    const chatbotInput = document.getElementById('chatbotInput');
    
    console.log('🔧 Setting up chatbot events...');
    
    // Toggle chatbot window
    chatbotButton.addEventListener('click', function(e) {
        e.stopPropagation();
        toggleChatbot();
    });
    
    closeChatBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        toggleChatbot();
    });
    
    // Clear conversation
    clearChatBtn.addEventListener('click', clearConversation);
    
    // Send message
    sendChatBtn.addEventListener('click', sendMessage);
    chatbotInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Prevent chatbot window clicks from propagating
    chatbotWindow.addEventListener('click', function(event) {
        event.stopPropagation();
    });
    
    // Close chatbot with Escape key
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape' && chatbotOpen) {
            console.log('⌨️ Escape key pressed, closing chatbot');
            closeChatbotWindow();
        }
    });
    
    // Close chatbot when clicking outside - using setTimeout to ensure it runs after other events
    document.addEventListener('click', function(event) {
        setTimeout(() => {
            if (chatbotOpen) {
                const clickedInsideChatbot = chatbotWindow && chatbotWindow.contains(event.target);
                const clickedOnButton = chatbotButton && chatbotButton.contains(event.target);
                
                if (!clickedInsideChatbot && !clickedOnButton) {
                    console.log('🖱️ Clicked outside chatbot, closing...');
                    closeChatbotWindow();
                }
            }
        }, 10);
    }, true); // Use capture phase
    
    console.log('✅ Chatbot events setup complete');
}

function closeChatbotWindow() {
    if (chatbotOpen) {
        console.log('🔒 Closing chatbot window');
        chatbotOpen = false;
        const chatbotWindow = document.getElementById('chatbotWindow');
        const chatbotButton = document.getElementById('chatbotButton');
        if (chatbotWindow) chatbotWindow.style.display = 'none';
        if (chatbotButton) chatbotButton.style.display = 'flex';
    }
}

function toggleChatbot() {
    chatbotOpen = !chatbotOpen;
    const chatbotWindow = document.getElementById('chatbotWindow');
    const chatbotButton = document.getElementById('chatbotButton');
    
    if (chatbotOpen) {
        chatbotWindow.style.display = 'flex';
        chatbotButton.style.display = 'none';
        document.getElementById('chatbotInput').focus();
    } else {
        chatbotWindow.style.display = 'none';
        chatbotButton.style.display = 'flex';
    }
}

async function sendMessage() {
    const input = document.getElementById('chatbotInput');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Add user message to chat
    addMessage(message, 'user');
    input.value = '';
    
    // Show typing indicator
    showTypingIndicator();
    
    try {
        // Send to API
        const response = await fetch('/api/chatbot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                session_id: sessionId
            })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator();
        
        if (data.success) {
            // Add bot response
            addMessage(data.response, 'bot');
            
            // Handle redirect if present
            if (data.redirect) {
                setTimeout(() => {
                    addMessage(`Redirecting you to ${data.redirect}...`, 'bot');
                    setTimeout(() => {
                        window.location.href = data.redirect;
                    }, 1000);
                }, 500);
            }
        } else {
            addMessage(data.response || 'Sorry, I encountered an error.', 'bot');
        }
        
    } catch (error) {
        removeTypingIndicator();
        addMessage('Sorry, I encountered a connection error. Please try again.', 'bot');
        console.error('Chatbot error:', error);
    }
}

function addMessage(text, type) {
    const messagesContainer = document.getElementById('chatbotMessages');
    const messageHTML = `
        <div class="chatbot-message ${type}-message">
            <div class="message-avatar">${type === 'user' ? '👤' : '🤖'}</div>
            <div class="message-content">
                <div class="message-text">${formatMessage(text)}</div>
                <div class="message-time">${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</div>
            </div>
        </div>
    `;
    
    messagesContainer.insertAdjacentHTML('beforeend', messageHTML);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function formatMessage(text) {
    // Convert markdown-style formatting to HTML
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    text = text.replace(/\n/g, '<br>');
    
    // Convert URLs to links
    text = text.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank">$1</a>');
    
    // Convert routes to clickable links
    text = text.replace(/(\/(advanced-scanner|scanner-detail\/[A-Z0-9:]+|symbol\/[A-Z0-9:]+))/g, 
        '<a href="$1" class="route-link">$1</a>');
    
    return text;
}

function showTypingIndicator() {
    const messagesContainer = document.getElementById('chatbotMessages');
    const typingHTML = `
        <div class="chatbot-message bot-message typing-indicator" id="typingIndicator">
            <div class="message-avatar">🤖</div>
            <div class="message-content">
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        </div>
    `;
    messagesContainer.insertAdjacentHTML('beforeend', typingHTML);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) {
        indicator.remove();
    }
}

async function clearConversation() {
    if (!confirm('Clear conversation history?')) return;
    
    try {
        await fetch(`/api/chatbot/clear?session_id=${sessionId}`, {
            method: 'POST'
        });
        
        // Clear messages
        const messagesContainer = document.getElementById('chatbotMessages');
        messagesContainer.innerHTML = `
            <div class="chatbot-message bot-message">
                <div class="message-avatar">🤖</div>
                <div class="message-content">
                    <div class="message-text">
                        Conversation cleared! How can I help you?
                    </div>
                </div>
            </div>
        `;
        
        // Generate new session ID
        sessionId = 'session_' + Date.now();
        
    } catch (error) {
        console.error('Error clearing conversation:', error);
    }
}
