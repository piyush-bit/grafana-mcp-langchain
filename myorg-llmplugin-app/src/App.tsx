import React, { useState, useRef, useEffect } from 'react';
import './App.css';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

function App() {
  const [query, setQuery] = useState('');
  const [uid, setUid] = useState<string | undefined>(undefined);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'assistant',
      content:
        "Hello! I'm your GenAI assistant. Ask me about your logs, dashboards, or any questions you have about your data.",
      timestamp: new Date(),
    },
  ]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!query.trim()) {
      return;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: query.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setQuery('');
    setLoading(true);

    try {
      console.log("cahnges has been done !!");
      
      const body = uid ? { query: userMessage.content, uid } : { query: userMessage.content };
      const res = await fetch('http://localhost:8020/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      const data = await res.json();

      if (data.uid && !uid) {
        setUid(data.uid);
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: data.response || 'Sorry, no answer returned.',
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: 'Sorry, an error occurred.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    });
  };

  return (
    <div className="chat-panel" style={{ width: '400px', height: '600px' }}>
      {/* Header */}
      <div className="chat-header">
        <div className="header-left">
          <div className="status-indicator"></div>
          <h2 className="header-title">GenAI Assistant</h2>
        </div>
        <div className="header-status">Connected</div>
      </div>

      {/* Messages */}
      <div className="chat-messages">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`message-container ${
              message.type === 'user' ? 'message-user-container' : 'message-assistant-container'
            }`}
          >
            <div
              className={`message-bubble ${
                message.type === 'user' ? 'message-user' : 'message-assistant'
              }`}
            >
              <div className="message-content">{message.content}</div>
              <div className="message-time">{formatTime(message.timestamp)}</div>
            </div>
          </div>
        ))}

        {loading && (
          <div className="message-container message-assistant-container">
            <div className="message-bubble message-assistant loading-message">
              <div className="typing-indicator">
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="chat-input-area">
        <div className="input-container">
          <div className="textarea-container">
            <textarea
              ref={textareaRef}
              rows={1}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about your logs, dashboards, or metrics..."
              className="chat-textarea"
              disabled={loading}
            />
            <div className="input-hint">Enter to send</div>
          </div>
          <button
            onClick={handleSend}
            disabled={loading || !query.trim()}
            className={`send-button ${
              loading || !query.trim() ? 'send-button-disabled' : 'send-button-active'
            }`}
          >
            <svg className="send-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
              />
            </svg>
          </button>
        </div>

        {/* Quick Actions */}
        <div className="quick-actions">
          {[
            { text: 'Show recent errors', emoji: '🚨' },
            { text: 'Analyze performance', emoji: '📊' },
            { text: 'Dashboard summary', emoji: '📋' },
            { text: 'Query help', emoji: '❓' },
          ].map((suggestion) => (
            <button
              key={suggestion.text}
              onClick={() => setQuery(suggestion.text)}
              className="quick-action-button"
              disabled={loading}
            >
              <span className="button-emoji">{suggestion.emoji}</span>
              <span>{suggestion.text}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;