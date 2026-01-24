import React, { useState, useEffect, useRef } from 'react';
import {
  Send, Upload, Image as ImageIcon, FileText, File, X, Loader,
  Bot, User, Download, Trash2, RefreshCw, Settings, Maximize2,
  Minimize2, Copy, Check, AlertCircle, Fish, Microscope, BarChart2,
  Camera, Paperclip, Zap, Sparkles, MessageSquare, History, ChevronDown,
  Info, TrendingUp, Activity
} from 'lucide-react';

const API_BASE = 'http://localhost:8000';
const API_TIMEOUT = 30000; // 30 seconds

const ChatbotPage = () => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [expandedView, setExpandedView] = useState(false);
  const [copiedId, setCopiedId] = useState(null);
  const [apiHealth, setApiHealth] = useState('checking');
  
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const chatContainerRef = useRef(null);
  const timeoutRef = useRef(null);

  // Check API health on mount
  useEffect(() => {
    checkApiHealth();
    const healthInterval = setInterval(checkApiHealth, 30000);
    return () => clearInterval(healthInterval);
  }, []);

  const checkApiHealth = async () => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      
      const response = await fetch(`${API_BASE}/health`, {
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      setApiHealth(response.ok ? 'healthy' : 'error');
    } catch (error) {
      setApiHealth('error');
    }
  };

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  // Welcome message
  useEffect(() => {
    if (messages.length === 0) {
      setMessages([{
        id: Date.now(),
        type: 'bot',
        content: "🐠 **Namaste! I'm MeenaSetu AI** - Your intelligent aquatic expert!\n\nI can help you with:\n\n🔍 **Species Identification** - Upload fish images\n🏥 **Disease Detection** - Get diagnosis & treatment\n📊 **Data Visualization** - Create charts & graphs\n💬 **Expert Answers** - Ask anything about aquaculture\n\nHow can I assist you today?",
        timestamp: new Date()
      }]);
    }
  }, []);

  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files);
    if (files.length === 0) return;

    const validFiles = files.filter(file => {
      const validTypes = [
        'image/jpeg', 'image/png', 'image/gif', 'image/bmp',
        'application/pdf', 'text/csv', 'text/plain', 'application/json'
      ];
      const maxSize = 50 * 1024 * 1024; // 50MB
      return validTypes.includes(file.type) && file.size <= maxSize;
    });

    if (validFiles.length === 0) {
      addMessage('bot', '❌ **Invalid Files**\n\nPlease upload: Images (JPG, PNG), PDF, CSV, TXT, or JSON\n\n📋 Maximum file size: 50MB', true);
      return;
    }

    setUploadedFiles(prev => [...prev, ...validFiles]);
    const fileNames = validFiles.map(f => f.name).join(', ');
    addMessage('user', `📎 Uploaded: ${fileNames}`);
  };

  const removeFile = (index) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const addMessage = (type, content, isError = false) => {
    const newMessage = {
      id: Date.now() + Math.random(),
      type,
      content,
      timestamp: new Date(),
      isError
    };
    setMessages(prev => [...prev, newMessage]);
  };

  const parseMarkdown = (text) => {
    return text.split('\n').map((line, idx) => {
      // Headers (###)
      if (line.match(/^###\s+/)) {
        return (
          <div key={idx} style={{ fontWeight: '700', fontSize: '1rem', marginTop: '0.75rem', marginBottom: '0.5rem' }}>
            {line.replace(/^###\s+/, '')}
          </div>
        );
      }

      // Bold text with ** **
      if (line.includes('**')) {
        const parts = line.split(/(\*\*[^*]+\*\*)/);
        return (
          <div key={idx} style={{ marginTop: idx > 0 ? '0.25rem' : 0 }}>
            {parts.map((part, i) => {
              if (part.startsWith('**') && part.endsWith('**')) {
                return (
                  <span key={i} style={{ fontWeight: '700', color: 'inherit' }}>
                    {part.replace(/\*\*/g, '')}
                  </span>
                );
              }
              return <span key={i}>{part}</span>;
            })}
          </div>
        );
      }

      // List items (numbered or bulleted)
      if (line.match(/^[\d]+\.\s/) || line.match(/^[-*+]\s/)) {
        return (
          <div key={idx} style={{ marginLeft: '1.5rem', marginTop: '0.25rem' }}>
            {line}
          </div>
        );
      }

      // Emoji list items
      if (line.match(/^[🔍🏥📊💬✅⚠️❌📎📄📦💊🐟🏥📋💊🎯🔧]/)) {
        return (
          <div key={idx} style={{ marginTop: '0.5rem', marginBottom: '0.25rem' }}>
            {line}
          </div>
        );
      }

      // Empty lines
      return line ? <div key={idx}>{line}</div> : <br key={idx} />;
    });
  };

  const fetchWithTimeout = (url, options = {}) => {
    return Promise.race([
      fetch(url, options),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Request timeout')), API_TIMEOUT)
      )
    ]);
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() && uploadedFiles.length === 0) return;

    const userMessage = inputMessage.trim();
    setInputMessage('');

    if (userMessage) {
      addMessage('user', userMessage);
    }

    setIsLoading(true);
    setIsTyping(true);

    try {
      let response;
      let result;

      // Case 1: Image classification/disease detection
      if (uploadedFiles.some(f => f.type.startsWith('image/'))) {
        const imageFile = uploadedFiles.find(f => f.type.startsWith('image/'));
        const formData = new FormData();
        formData.append('file', imageFile);
        formData.append('detect_disease', 'true');
        formData.append('description', userMessage || 'Analyze this fish image');

        try {
          response = await fetchWithTimeout(`${API_BASE}/classify/fish`, {
            method: 'POST',
            body: formData
          });

          if (!response.ok) {
            throw new Error(`Server error: ${response.status} ${response.statusText}`);
          }

          result = await response.json();

          if (!result || typeof result !== 'object') {
            throw new Error('Invalid response format from server');
          }

          if (result.status === 'success' && result.species) {
            let botResponse = `🐟 **Species Classification**\n\n`;
            botResponse += `✅ **Identified Species:** ${result.species || 'Unknown'}\n`;
            botResponse += `📊 **Confidence Level:** ${((result.confidence || 0) * 100).toFixed(1)}%\n\n`;

            // Top predictions
            if (Array.isArray(result.top3_predictions) && result.top3_predictions.length > 0) {
              botResponse += `**Top Predictions:**\n`;
              result.top3_predictions.slice(0, 3).forEach((pred, idx) => {
                const species = pred?.species || pred?.name || 'Unknown';
                const conf = ((pred?.confidence || 0) * 100).toFixed(1);
                botResponse += `${idx + 1}. ${species} - ${conf}%\n`;
              });
              botResponse += `\n`;
            }

            // Disease detection
            if (result.disease_detection && typeof result.disease_detection === 'object') {
              const dd = result.disease_detection;
              botResponse += `🏥 **Health Assessment**\n\n`;

              if (dd.is_healthy === true || dd.status === 'healthy') {
                botResponse += `✅ **Status:** Healthy - No disease detected\n`;
                botResponse += `📊 **Confidence:** ${((dd.confidence || 0) * 100).toFixed(1)}%\n`;
              } else if (dd.is_healthy === false || dd.primary_disease || dd.predicted_disease) {
                const disease = dd.primary_disease || dd.predicted_disease || 'Unknown Disease';
                botResponse += `⚠️ **Disease Detected:** ${disease}\n`;
                botResponse += `📊 **Confidence:** ${((dd.confidence || 0) * 100).toFixed(1)}%\n`;

                if (Array.isArray(result.recommendations) && result.recommendations.length > 0) {
                  botResponse += `\n💊 **Treatment Plan:**\n\n`;
                  result.recommendations.forEach((rec, idx) => {
                    botResponse += `${idx + 1}. ${rec}\n`;
                  });
                }
              }
            }

            addMessage('bot', botResponse);
          } else {
            const errorMsg = result?.message || result?.error || 'Could not classify the image. Please try another image.';
            addMessage('bot', `❌ **Classification Error**\n\n${errorMsg}`, true);
          }

          setUploadedFiles([]);
        } catch (error) {
          addMessage('bot', `❌ **Image Classification Failed**\n\n${error.message || 'Unable to process image'}`, true);
          setUploadedFiles([]);
        }
      }
      // Case 2: Document upload
      else if (uploadedFiles.length > 0) {
        const formData = new FormData();
        formData.append('file', uploadedFiles[0]);
        formData.append('process_immediately', 'true');

        try {
          response = await fetchWithTimeout(`${API_BASE}/upload/document`, {
            method: 'POST',
            body: formData
          });

          if (!response.ok) {
            throw new Error(`Server error: ${response.status} ${response.statusText}`);
          }

          result = await response.json();

          if (!result || typeof result !== 'object') {
            throw new Error('Invalid response format from server');
          }

          if (result.status === 'success') {
            const filename = result.filename || uploadedFiles[0].name;
            const chunks = result.chunks_created || result.chunks || 0;
            addMessage('bot', `✅ **Document Uploaded Successfully**\n\n📄 **File:** ${filename}\n📦 **Processed Chunks:** ${chunks}\n\n💬 You can now ask questions about this document!`);
          } else {
            const errorMsg = result?.message || 'Unknown error during upload';
            addMessage('bot', `❌ **Upload Failed**\n\n${errorMsg}`, true);
          }

          setUploadedFiles([]);
        } catch (error) {
          addMessage('bot', `❌ **Document Upload Failed**\n\n${error.message || 'Unable to upload document'}`, true);
          setUploadedFiles([]);
        }
      }
      // Case 3: Text query
      else {
        const queryData = {
          query: userMessage,
          generate_visualization: userMessage.toLowerCase().includes('chart') ||
            userMessage.toLowerCase().includes('graph') ||
            userMessage.toLowerCase().includes('plot')
        };

        try {
          response = await fetchWithTimeout(`${API_BASE}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(queryData)
          });

          if (!response.ok) {
            throw new Error(`Server error: ${response.status} ${response.statusText}`);
          }

          result = await response.json();

          if (!result || typeof result !== 'object') {
            throw new Error('Invalid response format from server');
          }

          if (result.answer) {
            let botResponse = result.answer;

            // Visualization
            if (result.visualization && result.visualization.file_path) {
              botResponse += `\n\n📊 **Visualization Generated**\n`;
              botResponse += `📥 **Download:** ${result.visualization.download_url || 'Check outputs folder'}`;
            }

            // Species classification
            if (result.species_classification && typeof result.species_classification === 'object') {
              const sc = result.species_classification;
              const species = sc.predicted_species || 'Unknown';
              const conf = ((sc.confidence || 0) * 100).toFixed(1);
              botResponse += `\n\n🐟 **Species Identified:** ${species} (${conf}%)`;
            }

            // Disease detection
            if (result.disease_detection && typeof result.disease_detection === 'object') {
              const dd = result.disease_detection;
              if (dd.status === 'success' || dd.is_healthy !== undefined) {
                if (dd.is_healthy) {
                  botResponse += `\n✅ **Health Status:** Healthy`;
                } else {
                  const disease = dd.predicted_disease || 'Unknown Disease';
                  botResponse += `\n⚠️ **Disease Found:** ${disease}`;
                }
              }
            }

            addMessage('bot', botResponse);
          } else {
            addMessage('bot', `❌ **No Response**\n\nThe server did not return an answer. Please check your query and try again.`, true);
          }
        } catch (error) {
          addMessage('bot', `❌ **Query Failed**\n\n${error.message || 'Unable to process query'}`, true);
        }
      }
    } catch (error) {
      console.error('Unexpected error:', error);
      addMessage('bot', `❌ **Unexpected Error**\n\n${error.message || 'Something went wrong'}\n\n🔗 Make sure backend is running at: ${API_BASE}`, true);
    } finally {
      setIsLoading(false);
      setIsTyping(false);
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !isLoading) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const copyToClipboard = (text, id) => {
    navigator.clipboard.writeText(text).catch(err => {
      console.error('Copy failed:', err);
    });
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const clearChat = () => {
    if (window.confirm('Clear all chat history?')) {
      setMessages([{
        id: Date.now(),
        type: 'bot',
        content: "🐠 Chat cleared! How can I help you?",
        timestamp: new Date()
      }]);
      setUploadedFiles([]);
    }
  };

  const formatTimestamp = (date) => {
    const now = new Date();
    const diff = now - date;
    const minutes = Math.floor(diff / 60000);

    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;

    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;

    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  const suggestedQueries = [
    { icon: <Fish size={16} />, text: "Identify this fish species", category: "Classification" },
    { icon: <Microscope size={16} />, text: "What diseases affect rohu fish?", category: "Disease" },
    { icon: <BarChart2 size={16} />, text: "Show fish population data as chart", category: "Visualization" },
    { icon: <Activity size={16} />, text: "Best practices for fish farming", category: "Aquaculture" }
  ];

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(to bottom, #f8fafc 0%, #e2e8f0 100%)',
      padding: '2rem',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
      {/* Header */}
      <div style={{
        marginBottom: '2rem',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '1rem'
      }}>
        <div>
          <h1 style={{
            fontSize: '3rem',
            fontWeight: '800',
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            marginBottom: '0.5rem',
            display: 'flex',
            alignItems: 'center',
            gap: '1rem'
          }}>
            <MessageSquare size={48} color="#667eea" />
            MeenaSetu AI Chatbot
          </h1>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
            <div style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.5rem 1rem',
              background: apiHealth === 'healthy' ? '#10b981' : '#ef4444',
              color: 'white',
              borderRadius: '25px',
              fontSize: '0.85rem',
              fontWeight: '700',
              boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
            }}>
              <div style={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                background: 'white',
                animation: apiHealth === 'healthy' ? 'pulse 2s infinite' : 'none'
              }} />
              {apiHealth === 'healthy' ? 'AI Ready' : 'Connecting...'}
            </div>
            <div style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.5rem 1rem',
              background: 'white',
              border: '2px solid #e2e8f0',
              borderRadius: '25px',
              fontSize: '0.85rem',
              fontWeight: '700',
              color: '#64748b'
            }}>
              <Sparkles size={14} />
              Powered by EfficientNet + RAG
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
          <button
            onClick={() => setShowHistory(!showHistory)}
            style={{
              padding: '0.75rem 1.5rem',
              borderRadius: '12px',
              border: '2px solid #3b82f6',
              background: showHistory ? '#3b82f6' : 'white',
              color: showHistory ? 'white' : '#3b82f6',
              cursor: 'pointer',
              fontWeight: '700',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              transition: 'all 0.3s ease'
            }}
          >
            <History size={18} />
            History
          </button>

          <button
            onClick={clearChat}
            style={{
              padding: '0.75rem 1.5rem',
              borderRadius: '12px',
              border: '2px solid #ef4444',
              background: 'white',
              color: '#ef4444',
              cursor: 'pointer',
              fontWeight: '700',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              transition: 'all 0.3s ease'
            }}
          >
            <Trash2 size={18} />
            Clear
          </button>

          <button
            onClick={() => setExpandedView(!expandedView)}
            style={{
              padding: '0.75rem 1.5rem',
              borderRadius: '12px',
              border: '2px solid #667eea',
              background: 'white',
              color: '#667eea',
              cursor: 'pointer',
              fontWeight: '700',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              transition: 'all 0.3s ease'
            }}
          >
            {expandedView ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
          </button>
        </div>
      </div>

      {/* Main Chat Container */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: showHistory ? '1fr 350px' : '1fr',
        gap: '2rem',
        height: expandedView ? 'calc(100vh - 200px)' : 'calc(100vh - 300px)'
      }}>
        {/* Chat Area */}
        <div style={{
          background: 'white',
          borderRadius: '24px',
          boxShadow: '0 8px 32px rgba(0,0,0,0.08)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden'
        }}>
          {/* Messages Container */}
          <div
            ref={chatContainerRef}
            style={{
              flex: 1,
              overflowY: 'auto',
              padding: '2rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '1.5rem'
            }}
          >
            {messages.map((message) => (
              <div
                key={message.id}
                style={{
                  display: 'flex',
                  gap: '1rem',
                  alignItems: 'flex-start',
                  justifyContent: message.type === 'user' ? 'flex-end' : 'flex-start'
                }}
              >
                {message.type === 'bot' && (
                  <div style={{
                    width: 42,
                    height: 42,
                    borderRadius: '12px',
                    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                    boxShadow: '0 4px 12px rgba(102, 126, 234, 0.4)'
                  }}>
                    <Bot size={24} color="white" />
                  </div>
                )}

                <div style={{
                  maxWidth: '75%',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.5rem'
                }}>
                  <div style={{
                    padding: '1.25rem',
                    borderRadius: '16px',
                    background: message.type === 'user'
                      ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
                      : message.isError
                      ? 'linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)'
                      : 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)',
                    color: message.type === 'user' ? 'white' : '#1e293b',
                    boxShadow: message.type === 'user'
                      ? '0 4px 12px rgba(102, 126, 234, 0.4)'
                      : '0 2px 8px rgba(0,0,0,0.08)',
                    wordBreak: 'break-word',
                    fontSize: '0.95rem',
                    lineHeight: 1.6
                  }}>
                    {parseMarkdown(message.content)}
                  </div>

                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.75rem',
                    fontSize: '0.75rem',
                    color: '#94a3b8',
                    paddingLeft: message.type === 'user' ? '0' : '0.5rem'
                  }}>
                    <span>{formatTimestamp(message.timestamp)}</span>
                    {message.type === 'bot' && (
                      <button
                        onClick={() => copyToClipboard(message.content, message.id)}
                        style={{
                          background: 'none',
                          border: 'none',
                          cursor: 'pointer',
                          padding: '0.25rem',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.25rem',
                          color: '#94a3b8',
                          transition: 'color 0.2s'
                        }}
                      >
                        {copiedId === message.id ? <Check size={14} /> : <Copy size={14} />}
                      </button>
                    )}
                  </div>
                </div>

                {message.type === 'user' && (
                  <div style={{
                    width: 42,
                    height: 42,
                    borderRadius: '12px',
                    background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                    boxShadow: '0 4px 12px rgba(16, 185, 129, 0.4)'
                  }}>
                    <User size={24} color="white" />
                  </div>
                )}
              </div>
            ))}

            {isTyping && (
              <div style={{
                display: 'flex',
                gap: '1rem',
                alignItems: 'flex-start'
              }}>
                <div style={{
                  width: 42,
                  height: 42,
                  borderRadius: '12px',
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  boxShadow: '0 4px 12px rgba(102, 126, 234, 0.4)'
                }}>
                  <Bot size={24} color="white" />
                </div>
                <div style={{
                  padding: '1.25rem',
                  borderRadius: '16px',
                  background: 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
                  display: 'flex',
                  gap: '0.5rem'
                }}>
                  <div style={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    background: '#667eea',
                    animation: 'bounce 1.4s infinite ease-in-out both',
                    animationDelay: '-0.32s'
                  }} />
                  <div style={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    background: '#667eea',
                    animation: 'bounce 1.4s infinite ease-in-out both',
                    animationDelay: '-0.16s'
                  }} />
                  <div style={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    background: '#667eea',
                    animation: 'bounce 1.4s infinite ease-in-out both'
                  }} />
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Suggested Queries */}
          {messages.length <= 1 && (
            <div style={{
              padding: '1rem 2rem',
              borderTop: '2px solid #f1f5f9',
              display: 'flex',
              gap: '0.75rem',
              flexWrap: 'wrap'
            }}>
              {suggestedQueries.map((query, idx) => (
                <button
                  key={idx}
                  onClick={() => setInputMessage(query.text)}
                  disabled={isLoading}
                  style={{
                    padding: '0.75rem 1rem',
                    borderRadius: '12px',
                    border: '2px solid #e2e8f0',
                    background: 'white',
                    cursor: isLoading ? 'not-allowed' : 'pointer',
                    fontSize: '0.85rem',
                    fontWeight: '600',
                    color: '#64748b',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    transition: 'all 0.3s ease',
                    opacity: isLoading ? 0.5 : 1
                  }}
                  onMouseEnter={(e) => {
                    if (!isLoading) {
                      e.currentTarget.style.borderColor = '#667eea';
                      e.currentTarget.style.color = '#667eea';
                      e.currentTarget.style.transform = 'translateY(-2px)';
                    }
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = '#e2e8f0';
                    e.currentTarget.style.color = '#64748b';
                    e.currentTarget.style.transform = 'translateY(0)';
                  }}
                >
                  {query.icon}
                  {query.text}
                </button>
              ))}
            </div>
          )}

          {/* File Upload Preview */}
          {uploadedFiles.length > 0 && (
            <div style={{
              padding: '1rem 2rem',
              borderTop: '2px solid #f1f5f9',
              background: '#fefce8'
            }}>
              <div style={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: '0.75rem'
              }}>
                {uploadedFiles.map((file, idx) => (
                  <div
                    key={idx}
                    style={{
                      padding: '0.75rem 1rem',
                      borderRadius: '12px',
                      background: 'white',
                      border: '2px solid #fbbf24',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.75rem',
                      fontSize: '0.85rem',
                      fontWeight: '600'
                    }}
                  >
                    {file.type.startsWith('image/') ? <ImageIcon size={18} color="#fbbf24" /> :
                      file.type === 'application/pdf' ? <FileText size={18} color="#ef4444" /> :
                      <File size={18} color="#3b82f6" />}
                    <span style={{ maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {file.name}
                    </span>
                    <button
                      onClick={() => removeFile(idx)}
                      disabled={isLoading}
                      style={{
                        background: 'none',
                        border: 'none',
                        cursor: isLoading ? 'not-allowed' : 'pointer',
                        padding: '0.25rem',
                        display: 'flex',
                        opacity: isLoading ? 0.5 : 1
                      }}
                    >
                      <X size={16} color="#64748b" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Input Area */}
          <div style={{
            padding: '1.5rem 2rem',
            borderTop: '2px solid #f1f5f9',
            background: 'white'
          }}>
            <div style={{
              display: 'flex',
              gap: '1rem',
              alignItems: 'flex-end'
            }}>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                multiple
                accept="image/*,.pdf,.csv,.txt,.json"
                disabled={isLoading}
                style={{ display: 'none' }}
              />

              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isLoading}
                style={{
                  padding: '1rem',
                  borderRadius: '12px',
                  border: '2px solid #667eea',
                  background: 'white',
                  color: '#667eea',
                  cursor: isLoading ? 'not-allowed' : 'pointer',
                  opacity: isLoading ? 0.5 : 1,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  transition: 'all 0.3s ease',
                  flexShrink: 0
                }}
                onMouseEnter={(e) => {
                  if (!isLoading) {
                    e.currentTarget.style.background = '#667eea';
                    e.currentTarget.style.color = 'white';
                  }
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'white';
                  e.currentTarget.style.color = '#667eea';
                }}
              >
                <Paperclip size={20} />
              </button>

              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about fish species, diseases, or upload an image..."
                disabled={isLoading}
                rows={3}
                style={{
                  flex: 1,
                  padding: '1rem',
                  borderRadius: '12px',
                  border: '2px solid #e2e8f0',
                  fontSize: '0.95rem',
                  fontFamily: 'inherit',
                  resize: 'none',
                  outline: 'none',
                  transition: 'border-color 0.3s ease'
                }}
                onFocus={(e) => e.currentTarget.style.borderColor = '#667eea'}
                onBlur={(e) => e.currentTarget.style.borderColor = '#e2e8f0'}
              />

              <button
                onClick={handleSendMessage}
                disabled={isLoading || (!inputMessage.trim() && uploadedFiles.length === 0)}
                style={{
                  padding: '1rem 1.5rem',
                  borderRadius: '12px',
                  border: 'none',
                  background: isLoading || (!inputMessage.trim() && uploadedFiles.length === 0)
                    ? '#cbd5e1'
                    : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  color: 'white',
                  cursor: isLoading || (!inputMessage.trim() && uploadedFiles.length === 0) ? 'not-allowed' : 'pointer',
                  fontWeight: '700',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  boxShadow: isLoading || (!inputMessage.trim() && uploadedFiles.length === 0)
                    ? 'none'
                    : '0 4px 12px rgba(102, 126, 234, 0.4)',
                  transition: 'all 0.3s ease',
                  flexShrink: 0
                }}
                onMouseEnter={(e) => {
                  if (!isLoading && (inputMessage.trim() || uploadedFiles.length > 0)) {
                    e.currentTarget.style.transform = 'translateY(-2px)';
                    e.currentTarget.style.boxShadow = '0 6px 16px rgba(102, 126, 234, 0.5)';
                  }
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  if (!isLoading && (inputMessage.trim() || uploadedFiles.length > 0)) {
                    e.currentTarget.style.boxShadow = '0 4px 12px rgba(102, 126, 234, 0.4)';
                  }
                }}
              >
                {isLoading ? <Loader size={20} style={{ animation: 'spin 1s linear infinite' }} /> : <Send size={20} />}
                {isLoading ? 'Processing...' : 'Send'}
              </button>
            </div>

            <div style={{
              marginTop: '0.75rem',
              fontSize: '0.75rem',
              color: '#94a3b8',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}>
              <Info size={14} />
              Upload images for species identification & disease detection, or ask questions about aquaculture
            </div>
          </div>
        </div>

        {/* History Sidebar */}
        {showHistory && (
          <div style={{
            background: 'white',
            borderRadius: '24px',
            boxShadow: '0 8px 32px rgba(0,0,0,0.08)',
            padding: '2rem',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: '1.5rem',
              paddingBottom: '1rem',
              borderBottom: '2px solid #f1f5f9'
            }}>
              <h3 style={{
                margin: 0,
                fontSize: '1.25rem',
                fontWeight: '700',
                color: '#1e293b'
              }}>
                Chat History
              </h3>
              <span style={{
                padding: '0.25rem 0.75rem',
                background: '#667eea20',
                color: '#667eea',
                borderRadius: '20px',
                fontSize: '0.75rem',
                fontWeight: '700'
              }}>
                {messages.length} messages
              </span>
            </div>

            <div style={{
              flex: 1,
              overflowY: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: '1rem'
            }}>
              {messages.slice().reverse().map((message) => (
                <div
                  key={message.id}
                  style={{
                    padding: '1rem',
                    borderRadius: '12px',
                    background: message.type === 'user' ? '#667eea15' : '#f8fafc',
                    border: '2px solid',
                    borderColor: message.type === 'user' ? '#667eea30' : '#e2e8f0',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'translateX(-4px)';
                    e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.1)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'translateX(0)';
                    e.currentTarget.style.boxShadow = 'none';
                  }}
                >
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    marginBottom: '0.5rem'
                  }}>
                    {message.type === 'user' ? (
                      <User size={16} color="#667eea" />
                    ) : (
                      <Bot size={16} color="#667eea" />
                    )}
                    <span style={{
                      fontSize: '0.75rem',
                      fontWeight: '700',
                      color: '#667eea'
                    }}>
                      {message.type === 'user' ? 'You' : 'MeenaSetu AI'}
                    </span>
                    <span style={{
                      fontSize: '0.7rem',
                      color: '#94a3b8',
                      marginLeft: 'auto'
                    }}>
                      {formatTimestamp(message.timestamp)}
                    </span>
                  </div>
                  <div style={{
                    fontSize: '0.85rem',
                    color: '#64748b',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    display: '-webkit-box',
                    WebkitLineClamp: 3,
                    WebkitBoxOrient: 'vertical',
                    lineHeight: 1.4
                  }}>
                    {message.content}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        @keyframes bounce {
          0%, 80%, 100% { transform: translateY(0); }
          40% { transform: translateY(-10px); }
        }
        ::-webkit-scrollbar {
          width: 8px;
          height: 8px;
        }
        ::-webkit-scrollbar-track {
          background: #f1f5f9;
          border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
          background: linear-gradient(135deg, #5568d3 0%, #653a8b 100%);
        }
      `}</style>
    </div>
  );
};

export default ChatbotPage;