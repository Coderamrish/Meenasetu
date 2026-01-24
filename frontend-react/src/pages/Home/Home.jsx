import React, { useState, useEffect } from 'react';
import {
  Activity,
  BarChart2,
  Database,
  MessageCircle,
  Fish,
  Microscope,
  TrendingUp,
  Zap,
  Shield,
  Cloud,
  Sparkles,
  Award,
  Users,
  ArrowRight,
  CheckCircle,
  Star,
  Droplet,
  Eye,
  Target,
  Brain,
  Layers,
  GitBranch,
  Wifi,
  Lock,
  Globe,
  Cpu,
  Server,
  Circle
} from 'lucide-react';

const GRADIENTS = {
  purple: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
  pink: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
  blue: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
  green: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
  orange: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
  teal: 'linear-gradient(135deg, #0ba360 0%, #3cba92 100%)',
  ocean: 'linear-gradient(135deg, #2e3192 0%, #1bffff 100%)',
  sunset: 'linear-gradient(135deg, #ee0979 0%, #ff6a00 100%)',
};

const Home = () => {
  const [currentTestimonial, setCurrentTestimonial] = useState(0);
  const [hoveredFeature, setHoveredFeature] = useState(null);
  const [stats, setStats] = useState({
    queries: 1247,
    species: 31,
    accuracy: 94,
    users: 2500
  });

  // Animated counter effect
  useEffect(() => {
    const interval = setInterval(() => {
      setStats(prev => ({
        queries: Math.min(prev.queries + Math.floor(Math.random() * 5), 9999),
        species: 31,
        accuracy: 94 + (Math.random() * 2 - 1),
        users: Math.min(prev.users + Math.floor(Math.random() * 10), 99999)
      }));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  // Testimonial rotation
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTestimonial(prev => (prev + 1) % 3);
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const systemStatus = [
    { title: 'AI Core', status: 'Operational', metric: '99.9%', label: 'Uptime', icon: <Brain size={32} />, color: '#4caf50', gradient: GRADIENTS.green },
    { title: 'Vector DB', status: 'Connected', metric: '1.2M', label: 'Documents', icon: <Database size={32} />, color: '#2196f3', gradient: GRADIENTS.blue },
    { title: 'ML Models', status: 'Ready', metric: '31', label: 'Species', icon: <Fish size={32} />, color: '#ff9800', gradient: GRADIENTS.orange },
    { title: 'API', status: 'v2.0.0', metric: '<100ms', label: 'Response', icon: <Server size={32} />, color: '#9c27b0', gradient: GRADIENTS.purple },
  ];

  const features = [
    {
      icon: <Fish size={48} />,
      title: 'Species Classification',
      description: 'Identify 31 different fish species using advanced EfficientNet CNN models with 94% accuracy.',
      gradient: GRADIENTS.purple,
      stats: '94% Accuracy',
      badge: 'AI-Powered'
    },
    {
      icon: <Microscope size={48} />,
      title: 'Disease Detection',
      description: 'Detect fish diseases early and receive AI-powered treatment recommendations instantly.',
      gradient: GRADIENTS.pink,
      stats: '6+ Diseases',
      badge: 'Real-time'
    },
    {
      icon: <MessageCircle size={48} />,
      title: 'Conversational AI',
      description: 'Ask questions and get intelligent answers with our RAG-based conversational system.',
      gradient: GRADIENTS.blue,
      stats: '24/7 Available',
      badge: 'Smart'
    },
    {
      icon: <BarChart2 size={48} />,
      title: 'Smart Analytics',
      description: 'Generate dynamic charts and visual analytics from your aquaculture data automatically.',
      gradient: GRADIENTS.green,
      stats: 'Auto-Generate',
      badge: 'Advanced'
    },
    {
      icon: <Database size={48} />,
      title: 'Document Intelligence',
      description: 'Upload PDFs, CSVs, and images for AI-powered data extraction and analysis.',
      gradient: GRADIENTS.orange,
      stats: '1.2M+ Docs',
      badge: 'Intelligent'
    },
    {
      icon: <TrendingUp size={48} />,
      title: 'Performance Tracking',
      description: 'Monitor farm performance, track trends, and make data-driven decisions.',
      gradient: GRADIENTS.teal,
      stats: 'Live Tracking',
      badge: 'Real-time'
    },
  ];

  const whyChoose = [
    {
      icon: <Shield size={40} />,
      title: 'Enterprise Security',
      description: 'Bank-grade encryption and secure data handling',
      color: '#667eea'
    },
    {
      icon: <Zap size={40} />,
      title: 'Lightning Fast',
      description: 'Real-time predictions under 100ms response time',
      color: '#f5576c'
    },
    {
      icon: <Cloud size={40} />,
      title: 'Cloud Native',
      description: '99.9% uptime with scalable infrastructure',
      color: '#00f2fe'
    },
    {
      icon: <Sparkles size={40} />,
      title: 'AI-Powered',
      description: 'State-of-the-art machine learning models',
      color: '#38f9d7'
    },
  ];

  const capabilities = [
    { text: 'RAG-based Q&A with context-aware responses', icon: <MessageCircle size={20} /> },
    { text: 'CNN-based disease detection with treatment suggestions', icon: <Microscope size={20} /> },
    { text: 'EfficientNet image classification for 31 fish species', icon: <Fish size={20} /> },
    { text: 'ChromaDB vector search for semantic retrieval', icon: <Database size={20} /> },
    { text: 'Multi-turn conversation with memory', icon: <Brain size={20} /> },
    { text: 'Automatic visualization generation', icon: <BarChart2 size={20} /> },
    { text: 'Real-time performance monitoring', icon: <Activity size={20} /> },
    { text: 'Batch processing for large datasets', icon: <Layers size={20} /> },
  ];

  const testimonials = [
    {
      name: 'Dr. Rajesh Kumar',
      role: 'Aquaculture Researcher',
      company: 'Indian Institute of Technology',
      image: '👨‍🔬',
      quote: 'MeenaSetu AI has revolutionized our research. The accuracy of species identification and disease detection is remarkable!',
      rating: 5
    },
    {
      name: 'Priya Sharma',
      role: 'Fish Farm Owner',
      company: 'Blue Ocean Fisheries',
      image: '👩‍💼',
      quote: 'This platform helped us prevent a major disease outbreak. The AI recommendations saved thousands of fish and our business.',
      rating: 5
    },
    {
      name: 'Mohammed Ali',
      role: 'Marine Biologist',
      company: 'Coastal Conservation Society',
      image: '👨‍🎓',
      quote: 'An invaluable tool for biodiversity research. The conversational AI makes complex data accessible to everyone.',
      rating: 5
    }
  ];

  const techStack = [
    { name: 'EfficientNet-B0', desc: 'Image Classification' },
    { name: 'LangChain', desc: 'RAG Framework' },
    { name: 'ChromaDB', desc: 'Vector Database' },
    { name: 'Groq LLM', desc: 'AI Processing' },
    { name: 'PyTorch', desc: 'Deep Learning' },
    { name: 'FastAPI', desc: 'Backend API' },
  ];

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(to bottom, #f8fafc 0%, #e2e8f0 100%)' }}>
      {/* Hero Section */}
      <div
        style={{
          background: GRADIENTS.ocean,
          position: 'relative',
          overflow: 'hidden',
          padding: '6rem 2rem',
          color: 'white',
        }}
      >
        {/* Animated background elements */}
        <div
          style={{
            position: 'absolute',
            top: '-10%',
            left: '-5%',
            width: '400px',
            height: '400px',
            borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%)',
            animation: 'float 8s ease-in-out infinite',
          }}
        />
        <div
          style={{
            position: 'absolute',
            bottom: '-10%',
            right: '-5%',
            width: '500px',
            height: '500px',
            borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 70%)',
            animation: 'float 10s ease-in-out infinite',
            animationDelay: '2s',
          }}
        />

        <div style={{ maxWidth: '1200px', margin: '0 auto', position: 'relative', zIndex: 1 }}>
          {/* Badge */}
          <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
            <div
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.75rem 1.5rem',
                background: 'rgba(255, 255, 255, 0.2)',
                backdropFilter: 'blur(10px)',
                borderRadius: '50px',
                border: '2px solid rgba(255, 255, 255, 0.3)',
                fontSize: '0.95rem',
                fontWeight: '700',
                boxShadow: '0 8px 32px rgba(0,0,0,0.1)',
              }}
            >
              <Sparkles size={20} />
              AI-Powered Aquaculture Intelligence Platform
            </div>
          </div>

          {/* Main Heading */}
          <h1
            style={{
              fontSize: '5rem',
              fontWeight: '900',
              textAlign: 'center',
              marginBottom: '1.5rem',
              letterSpacing: '-2px',
              textShadow: '0 4px 20px rgba(0,0,0,0.2)',
              animation: 'fadeInUp 0.8s ease',
            }}
          >
            MeenaSetu
          </h1>

          <p
            style={{
              fontSize: '2rem',
              textAlign: 'center',
              marginBottom: '1rem',
              opacity: 0.95,
              fontWeight: '300',
              animation: 'fadeInUp 1s ease',
            }}
          >
            Intelligent Aquatic Expert
          </p>

          <p
            style={{
              fontSize: '1.3rem',
              textAlign: 'center',
              marginBottom: '3rem',
              opacity: 0.9,
              maxWidth: '900px',
              margin: '0 auto 3rem',
              lineHeight: 1.6,
              animation: 'fadeInUp 1.2s ease',
            }}
          >
            Transform your aquaculture operations with AI-powered insights, disease detection, and smart analytics.
            Make data-driven decisions with confidence.
          </p>

          {/* CTA Buttons */}
          <div style={{ display: 'flex', gap: '1.5rem', justifyContent: 'center', flexWrap: 'wrap' }}>
            <button
              style={{
                padding: '1.25rem 3rem',
                fontSize: '1.2rem',
                fontWeight: '700',
                borderRadius: '16px',
                border: 'none',
                background: 'white',
                color: '#2e3192',
                cursor: 'pointer',
                boxShadow: '0 8px 24px rgba(0,0,0,0.2)',
                transition: 'all 0.3s ease',
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-4px)';
                e.currentTarget.style.boxShadow = '0 12px 32px rgba(0,0,0,0.3)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 8px 24px rgba(0,0,0,0.2)';
              }}
            >
              <MessageCircle size={24} />
              Start Conversation
            </button>
            <button
              style={{
                padding: '1.25rem 3rem',
                fontSize: '1.2rem',
                fontWeight: '700',
                borderRadius: '16px',
                border: '3px solid white',
                background: 'transparent',
                color: 'white',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.15)';
                e.currentTarget.style.transform = 'translateY(-4px)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'transparent';
                e.currentTarget.style.transform = 'translateY(0)';
              }}
            >
              <Database size={24} />
              Upload Data
            </button>
          </div>

          {/* Live Stats */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '2rem',
              marginTop: '4rem',
            }}
          >
            {[
              { value: stats.queries.toLocaleString(), label: 'Queries Processed', icon: <Activity size={24} /> },
              { value: stats.species, label: 'Fish Species', icon: <Fish size={24} /> },
              { value: `${stats.accuracy.toFixed(1)}%`, label: 'AI Accuracy', icon: <Target size={24} /> },
              { value: stats.users.toLocaleString(), label: 'Active Users', icon: <Users size={24} /> },
            ].map((stat, idx) => (
              <div
                key={idx}
                style={{
                  background: 'rgba(255, 255, 255, 0.15)',
                  backdropFilter: 'blur(10px)',
                  borderRadius: '20px',
                  padding: '2rem',
                  textAlign: 'center',
                  border: '2px solid rgba(255, 255, 255, 0.25)',
                  transition: 'all 0.3s ease',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-8px)';
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.25)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.15)';
                }}
              >
                <div style={{ marginBottom: '0.75rem' }}>{stat.icon}</div>
                <div style={{ fontSize: '2.5rem', fontWeight: '800', marginBottom: '0.5rem' }}>{stat.value}</div>
                <div style={{ fontSize: '0.95rem', opacity: 0.95 }}>{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* System Status Cards */}
      <div style={{ maxWidth: '1200px', margin: '-4rem auto 4rem', padding: '0 2rem', position: 'relative', zIndex: 10 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '2rem' }}>
          {systemStatus.map((card, idx) => (
            <div
              key={idx}
              style={{
                background: 'white',
                borderRadius: '24px',
                padding: '2rem',
                boxShadow: '0 8px 32px rgba(0,0,0,0.12)',
                transition: 'all 0.4s ease',
                position: 'relative',
                overflow: 'hidden',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-12px)';
                e.currentTarget.style.boxShadow = '0 16px 48px rgba(0,0,0,0.2)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 8px 32px rgba(0,0,0,0.12)';
              }}
            >
              {/* Background gradient */}
              <div
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  height: '4px',
                  background: card.gradient,
                }}
              />

              <div
                style={{
                  width: '70px',
                  height: '70px',
                  borderRadius: '18px',
                  background: card.gradient,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                  marginBottom: '1.5rem',
                  boxShadow: `0 8px 24px ${card.color}40`,
                }}
              >
                {card.icon}
              </div>

              <h3 style={{ margin: 0, fontSize: '1.3rem', fontWeight: '700', color: '#1e293b', marginBottom: '0.75rem' }}>
                {card.title}
              </h3>

              <div
                style={{
                  display: 'inline-block',
                  padding: '0.5rem 1rem',
                  borderRadius: '20px',
                  background: `${card.color}20`,
                  color: card.color,
                  fontSize: '0.85rem',
                  fontWeight: '700',
                  marginBottom: '1rem',
                }}
              >
                {card.status}
              </div>

              <div style={{ borderTop: '2px solid #e2e8f0', paddingTop: '1rem', marginTop: '1rem' }}>
                <div style={{ fontSize: '2rem', fontWeight: '800', color: card.color, marginBottom: '0.25rem' }}>
                  {card.metric}
                </div>
                <div style={{ fontSize: '0.9rem', color: '#64748b' }}>{card.label}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Features Section */}
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '4rem 2rem' }}>
        <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.5rem 1.25rem',
              background: 'linear-gradient(135deg, #667eea20 0%, #764ba220 100%)',
              borderRadius: '50px',
              fontSize: '0.85rem',
              fontWeight: '700',
              color: '#667eea',
              marginBottom: '1.5rem',
            }}
          >
            <Star size={16} />
            PLATFORM FEATURES
          </div>
          <h2
            style={{
              fontSize: '3.5rem',
              fontWeight: '800',
              color: '#1e293b',
              marginBottom: '1rem',
              letterSpacing: '-1px',
            }}
          >
            Everything You Need
          </h2>
          <p style={{ fontSize: '1.2rem', color: '#64748b', maxWidth: '700px', margin: '0 auto' }}>
            Comprehensive AI-powered tools designed for modern aquaculture management
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '2.5rem' }}>
          {features.map((feature, idx) => (
            <div
              key={idx}
              onMouseEnter={() => setHoveredFeature(idx)}
              onMouseLeave={() => setHoveredFeature(null)}
              style={{
                background: 'white',
                borderRadius: '28px',
                padding: '2.5rem',
                boxShadow: '0 8px 32px rgba(0,0,0,0.08)',
                transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
                position: 'relative',
                overflow: 'hidden',
                transform: hoveredFeature === idx ? 'translateY(-12px)' : 'translateY(0)',
                border: hoveredFeature === idx ? '2px solid #667eea40' : '2px solid transparent',
              }}
            >
              {/* Gradient overlay on hover */}
              <div
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  background: feature.gradient,
                  opacity: hoveredFeature === idx ? 0.05 : 0,
                  transition: 'opacity 0.4s ease',
                }}
              />

              <div style={{ position: 'relative', zIndex: 1 }}>
                {/* Icon */}
                <div
                  style={{
                    width: '90px',
                    height: '90px',
                    borderRadius: '22px',
                    background: feature.gradient,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'white',
                    marginBottom: '2rem',
                    boxShadow: '0 12px 32px rgba(102, 126, 234, 0.3)',
                    transform: hoveredFeature === idx ? 'scale(1.1) rotate(5deg)' : 'scale(1) rotate(0deg)',
                    transition: 'all 0.4s ease',
                  }}
                >
                  {feature.icon}
                </div>

                {/* Badge */}
                <div
                  style={{
                    display: 'inline-block',
                    padding: '0.4rem 0.9rem',
                    borderRadius: '20px',
                    background: feature.gradient,
                    color: 'white',
                    fontSize: '0.75rem',
                    fontWeight: '700',
                    marginBottom: '1.5rem',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                  }}
                >
                  {feature.badge}
                </div>

                {/* Content */}
                <h3 style={{ margin: 0, fontSize: '1.6rem', fontWeight: '700', color: '#1e293b', marginBottom: '1rem' }}>
                  {feature.title}
                </h3>

                <p style={{ margin: 0, fontSize: '1rem', color: '#64748b', lineHeight: 1.7, marginBottom: '1.5rem' }}>
                  {feature.description}
                </p>

                {/* Stats */}
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '1rem',
                    background: '#f8fafc',
                    borderRadius: '12px',
                    marginBottom: '1.5rem',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <TrendingUp size={18} color="#10b981" />
                    <span style={{ fontSize: '0.95rem', fontWeight: '600', color: '#1e293b' }}>{feature.stats}</span>
                  </div>
                </div>

                {/* CTA Button */}
                <button
                  style={{
                    width: '100%',
                    padding: '1rem',
                    borderRadius: '12px',
                    border: 'none',
                    background: hoveredFeature === idx ? feature.gradient : '#f8fafc',
                    color: hoveredFeature === idx ? 'white' : '#667eea',
                    fontWeight: '700',
                    fontSize: '1rem',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '0.5rem',
                  }}
                >
                  Explore Feature
                  <ArrowRight size={18} />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Why Choose Section */}
      <div style={{ background: 'white', padding: '5rem 2rem', marginTop: '4rem' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
            <h2 style={{ fontSize: '3rem', fontWeight: '800', color: '#1e293b', marginBottom: '1rem' }}>
              Why Choose MeenaSetu?
            </h2>
            <p style={{ fontSize: '1.2rem', color: '#64748b', maxWidth: '700px', margin: '0 auto' }}>
              Built with cutting-edge technology for reliability and performance
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '3rem' }}>
            {whyChoose.map((item, idx) => (
              <div
                key={idx}
                style={{
                  textAlign: 'center',
                  transition: 'all 0.3s ease',
                }}
                onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.05)'}
                onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
              >
                <div
                  style={{
                    width: '100px',
                    height: '100px',
                    borderRadius: '50%',
                    background: GRADIENTS.purple,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'white',
                    margin: '0 auto 1.5rem',
                    boxShadow: '0 12px 32px rgba(102, 126, 234, 0.4)',
                  }}
                >
                  {item.icon}
                </div>
                <h3 style={{ margin: 0, fontSize: '1.4rem', fontWeight: '700', color: '#1e293b', marginBottom: '0.75rem' }}>
                  {item.title}
                </h3>
                <p style={{ margin: 0, fontSize: '1rem', color: '#64748b', lineHeight: 1.6 }}>{item.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Capabilities & Tech Stack */}
      <div style={{ maxWidth: '1200px', margin: '5rem auto', padding: '0 2rem' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))', gap: '3rem' }}>
          {/* Capabilities */}
          <div
            style={{
              background: 'white',
              borderRadius: '28px',
              padding: '3rem',
              boxShadow: '0 8px 32px rgba(0,0,0,0.08)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '2rem' }}>
              <div
                style={{
                  width: '60px',
                  height: '60px',
                  borderRadius: '16px',
                  background: GRADIENTS.blue,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                }}
              >
                <Cpu size={32} />
              </div>
              <h3 style={{ margin: 0, fontSize: '2rem', fontWeight: '700', color: '#1e293b' }}>Platform Capabilities</h3>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {capabilities.map((cap, idx) => (
                <div
                  key={idx}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '1rem',
                    padding: '1rem',
                    borderRadius: '12px',
                    background: '#f8fafc',
                    transition: 'all 0.3s ease',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = '#667eea10';
                    e.currentTarget.style.transform = 'translateX(8px)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = '#f8fafc';
                    e.currentTarget.style.transform = 'translateX(0)';
                  }}
                >
                  <div
                    style={{
                      width: '40px',
                      height: '40px',
                      borderRadius: '10px',
                      background: GRADIENTS.blue,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white',
                      flexShrink: 0,
                    }}
                  >
                    {cap.icon}
                  </div>
                  <span style={{ fontSize: '1rem', fontWeight: '500', color: '#1e293b' }}>{cap.text}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Tech Stack */}
          <div
            style={{
              background: 'white',
              borderRadius: '28px',
              padding: '3rem',
              boxShadow: '0 8px 32px rgba(0,0,0,0.08)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '2rem' }}>
              <div
                style={{
                  width: '60px',
                  height: '60px',
                  borderRadius: '16px',
                  background: GRADIENTS.purple,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                }}
              >
                <Layers size={32} />
              </div>
              <h3 style={{ margin: 0, fontSize: '2rem', fontWeight: '700', color: '#1e293b' }}>Powered By</h3>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1.5rem' }}>
              {techStack.map((tech, idx) => (
                <div
                  key={idx}
                  style={{
                    padding: '1.5rem',
                    borderRadius: '16px',
                    background: 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)',
                    border: '2px solid #e2e8f0',
                    transition: 'all 0.3s ease',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'translateY(-4px)';
                    e.currentTarget.style.borderColor = '#667eea';
                    e.currentTarget.style.boxShadow = '0 8px 24px rgba(102, 126, 234, 0.2)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.borderColor = '#e2e8f0';
                    e.currentTarget.style.boxShadow = 'none';
                  }}
                >
                  <div style={{ fontSize: '1.2rem', fontWeight: '700', color: '#1e293b', marginBottom: '0.5rem' }}>
                    {tech.name}
                  </div>
                  <div style={{ fontSize: '0.9rem', color: '#64748b' }}>{tech.desc}</div>
                </div>
              ))}
            </div>

            {/* Quick Stats */}
            <div
              style={{
                marginTop: '2rem',
                padding: '1.5rem',
                borderRadius: '16px',
                background: GRADIENTS.purple,
                color: 'white',
                textAlign: 'center',
              }}
            >
              <div style={{ fontSize: '2.5rem', fontWeight: '800', marginBottom: '0.5rem' }}>99.9%</div>
              <div style={{ fontSize: '1rem', opacity: 0.95 }}>System Uptime Guarantee</div>
            </div>
          </div>
        </div>
      </div>

      {/* Testimonials */}
      <div style={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', padding: '5rem 2rem' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <div style={{ textAlign: 'center', marginBottom: '4rem', color: 'white' }}>
            <h2 style={{ fontSize: '3rem', fontWeight: '800', marginBottom: '1rem' }}>What Our Users Say</h2>
            <p style={{ fontSize: '1.2rem', opacity: 0.95 }}>Trusted by researchers, farmers, and marine biologists worldwide</p>
          </div>

          <div style={{ position: 'relative' }}>
            {testimonials.map((testimonial, idx) => (
              <div
                key={idx}
                style={{
                  display: idx === currentTestimonial ? 'block' : 'none',
                  animation: 'fadeIn 0.5s ease',
                }}
              >
                <div
                  style={{
                    background: 'rgba(255, 255, 255, 0.15)',
                    backdropFilter: 'blur(20px)',
                    borderRadius: '28px',
                    padding: '3rem',
                    border: '2px solid rgba(255, 255, 255, 0.25)',
                    maxWidth: '800px',
                    margin: '0 auto',
                  }}
                >
                  <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '2rem', justifyContent: 'center' }}>
                    {[...Array(testimonial.rating)].map((_, i) => (
                      <Star key={i} size={24} fill="#fbbf24" color="#fbbf24" />
                    ))}
                  </div>

                  <p
                    style={{
                      fontSize: '1.5rem',
                      color: 'white',
                      textAlign: 'center',
                      lineHeight: 1.8,
                      marginBottom: '2rem',
                      fontStyle: 'italic',
                    }}
                  >
                    "{testimonial.quote}"
                  </p>

                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '1rem' }}>
                    <div
                      style={{
                        width: '60px',
                        height: '60px',
                        borderRadius: '50%',
                        background: 'rgba(255, 255, 255, 0.25)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '2rem',
                      }}
                    >
                      {testimonial.image}
                    </div>
                    <div style={{ color: 'white' }}>
                      <div style={{ fontSize: '1.2rem', fontWeight: '700' }}>{testimonial.name}</div>
                      <div style={{ fontSize: '1rem', opacity: 0.9 }}>
                        {testimonial.role}, {testimonial.company}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}

            {/* Dots */}
            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', marginTop: '2rem' }}>
              {testimonials.map((_, idx) => (
                <button
                  key={idx}
                  onClick={() => setCurrentTestimonial(idx)}
                  style={{
                    width: currentTestimonial === idx ? '40px' : '12px',
                    height: '12px',
                    borderRadius: '6px',
                    border: 'none',
                    background: currentTestimonial === idx ? 'white' : 'rgba(255, 255, 255, 0.4)',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                  }}
                />
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Integration Section */}
      <div style={{ maxWidth: '1200px', margin: '5rem auto', padding: '0 2rem' }}>
        <div
          style={{
            background: 'white',
            borderRadius: '32px',
            padding: '4rem',
            boxShadow: '0 12px 48px rgba(0,0,0,0.1)',
            textAlign: 'center',
          }}
        >
          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.5rem 1.25rem',
              background: 'linear-gradient(135deg, #43e97b20 0%, #38f9d720 100%)',
              borderRadius: '50px',
              fontSize: '0.85rem',
              fontWeight: '700',
              color: '#43e97b',
              marginBottom: '2rem',
            }}
          >
            <Wifi size={16} />
            SEAMLESS INTEGRATION
          </div>

          <h2 style={{ fontSize: '3rem', fontWeight: '800', color: '#1e293b', marginBottom: '1rem' }}>
            Easy to Integrate, Powerful to Use
          </h2>
          <p style={{ fontSize: '1.2rem', color: '#64748b', marginBottom: '3rem', maxWidth: '700px', margin: '0 auto 3rem' }}>
            RESTful API with comprehensive documentation. Get started in minutes with our SDKs and libraries.
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '2rem', marginBottom: '3rem' }}>
            {[
              { icon: <Globe size={32} />, title: 'REST API', desc: 'Full-featured API' },
              { icon: <Lock size={32} />, title: 'Secure', desc: 'Enterprise-grade' },
              { icon: <Zap size={32} />, title: 'Fast', desc: '<100ms response' },
              { icon: <Server size={32} />, title: 'Reliable', desc: '99.9% uptime' },
            ].map((item, idx) => (
              <div key={idx}>
                <div
                  style={{
                    width: '70px',
                    height: '70px',
                    borderRadius: '18px',
                    background: GRADIENTS.green,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'white',
                    margin: '0 auto 1rem',
                  }}
                >
                  {item.icon}
                </div>
                <div style={{ fontSize: '1.2rem', fontWeight: '700', color: '#1e293b', marginBottom: '0.5rem' }}>
                  {item.title}
                </div>
                <div style={{ fontSize: '0.95rem', color: '#64748b' }}>{item.desc}</div>
              </div>
            ))}
          </div>

          <button
            style={{
              padding: '1.25rem 3rem',
              fontSize: '1.1rem',
              fontWeight: '700',
              borderRadius: '16px',
              border: 'none',
              background: GRADIENTS.green,
              color: 'white',
              cursor: 'pointer',
              boxShadow: '0 8px 24px rgba(67, 233, 123, 0.4)',
              transition: 'all 0.3s ease',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.75rem',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-4px)';
              e.currentTarget.style.boxShadow = '0 12px 32px rgba(67, 233, 123, 0.5)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = '0 8px 24px rgba(67, 233, 123, 0.4)';
            }}
          >
            View API Documentation
            <ArrowRight size={20} />
          </button>
        </div>
      </div>

      {/* Final CTA */}
      <div style={{ background: 'linear-gradient(135deg, #0ba360 0%, #3cba92 100%)', padding: '6rem 2rem', position: 'relative', overflow: 'hidden' }}>
        {/* Background elements */}
        <div
          style={{
            position: 'absolute',
            top: '-10%',
            right: '-5%',
            width: '400px',
            height: '400px',
            borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%)',
          }}
        />

        <div style={{ maxWidth: '1200px', margin: '0 auto', textAlign: 'center', position: 'relative', zIndex: 1, color: 'white' }}>
          <div
            style={{
              width: '100px',
              height: '100px',
              borderRadius: '50%',
              background: 'rgba(255, 255, 255, 0.25)',
              backdropFilter: 'blur(10px)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 2rem',
            }}
          >
            <Award size={50} />
          </div>

          <h2 style={{ fontSize: '3.5rem', fontWeight: '800', marginBottom: '1.5rem', textShadow: '0 4px 20px rgba(0,0,0,0.2)' }}>
            Ready to Transform Your Aquaculture?
          </h2>

          <p style={{ fontSize: '1.4rem', marginBottom: '3rem', opacity: 0.95, maxWidth: '800px', margin: '0 auto 3rem', lineHeight: 1.6 }}>
            Join thousands of farmers and researchers worldwide in leveraging AI for smarter, more sustainable fisheries.
            Start making data-driven decisions today with MeenaSetu.
          </p>

          <div style={{ display: 'flex', gap: '1.5rem', justifyContent: 'center', flexWrap: 'wrap' }}>
            <button
              style={{
                padding: '1.5rem 3.5rem',
                fontSize: '1.3rem',
                fontWeight: '700',
                borderRadius: '16px',
                border: 'none',
                background: 'white',
                color: '#0ba360',
                cursor: 'pointer',
                boxShadow: '0 8px 24px rgba(0,0,0,0.2)',
                transition: 'all 0.3s ease',
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-4px) scale(1.05)';
                e.currentTarget.style.boxShadow = '0 16px 40px rgba(0,0,0,0.3)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0) scale(1)';
                e.currentTarget.style.boxShadow = '0 8px 24px rgba(0,0,0,0.2)';
              }}
            >
              Start Using MeenaSetu AI
              <ArrowRight size={24} />
            </button>

            <button
              style={{
                padding: '1.5rem 3.5rem',
                fontSize: '1.3rem',
                fontWeight: '700',
                borderRadius: '16px',
                border: '3px solid white',
                background: 'transparent',
                color: 'white',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.15)';
                e.currentTarget.style.transform = 'translateY(-4px)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'transparent';
                e.currentTarget.style.transform = 'translateY(0)';
              }}
            >
              Learn More
            </button>
          </div>

          {/* Trust Indicators */}
          <div
            style={{
              marginTop: '4rem',
              padding: '2rem',
              borderRadius: '20px',
              background: 'rgba(255, 255, 255, 0.15)',
              backdropFilter: 'blur(10px)',
              border: '2px solid rgba(255, 255, 255, 0.25)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap', gap: '2rem' }}>
              {[
                { icon: <CheckCircle size={24} />, text: '2,500+ Active Users' },
                { icon: <Shield size={24} />, text: 'Enterprise Security' },
                { icon: <Award size={24} />, text: '94% AI Accuracy' },
                { icon: <Users size={24} />, text: '24/7 Support' },
              ].map((item, idx) => (
                <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  {item.icon}
                  <span style={{ fontSize: '1rem', fontWeight: '600' }}>{item.text}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div style={{ background: '#1e293b', color: 'white', padding: '3rem 2rem', textAlign: 'center' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <div style={{ fontSize: '2rem', fontWeight: '800', marginBottom: '1rem' }}>MeenaSetu</div>
          <p style={{ opacity: 0.8, marginBottom: '2rem' }}>Intelligent Aquatic Expert Platform</p>
          <div style={{ borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '2rem', opacity: 0.6 }}>
            © 2026 MeenaSetu AI. All rights reserved. | Powered by AI for a sustainable future.
          </div>
        </div>
      </div>

      <style>{`
        @keyframes float {
          0%, 100% { 
            transform: translate(0, 0) scale(1);
            opacity: 0.5;
          }
          50% { 
            transform: translate(-20px, -20px) scale(1.1);
            opacity: 0.8;
          }
        }
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
      `}</style>
    </div>
  );
};

export default Home;