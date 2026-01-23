import React, { useState, useEffect, useRef } from 'react';
import { 
  TrendingUp, TrendingDown, Award, Users, Zap, Target,
  Download, Share2, ArrowUp, PlayCircle, X, ChevronDown,
  Linkedin, Github, Mail, CheckCircle, Star, Code, Database,
  Cloud, Cpu, Globe, Shield, BarChart2, Activity, Rocket,
  Lightbulb, Heart, Eye, MousePointer
} from 'lucide-react';

const About = () => {
  const [scrollProgress, setScrollProgress] = useState(0);
  const [stats, setStats] = useState({ users: 0, predictions: 0, accuracy: 0, papers: 0 });
  const [videoOpen, setVideoOpen] = useState(false);
  const [activeSection, setActiveSection] = useState('hero');
  const [showScrollTop, setShowScrollTop] = useState(false);

  // Scroll progress handler
  useEffect(() => {
    const handleScroll = () => {
      const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
      const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      setScrollProgress((winScroll / height) * 100);
      setShowScrollTop(winScroll > 500);

      // Update active section
      const sections = ['hero', 'mission', 'features', 'creator', 'tech', 'journey', 'team'];
      const current = sections.find(section => {
        const element = document.getElementById(section);
        if (element) {
          const rect = element.getBoundingClientRect();
          return rect.top <= 100 && rect.bottom >= 100;
        }
        return false;
      });
      if (current) setActiveSection(current);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Animated counters
  useEffect(() => {
    const duration = 2000;
    const steps = 60;
    const increment = (target, key) => {
      let start = 0;
      const step = target / steps;
      const timer = setInterval(() => {
        start += step;
        if (start >= target) {
          setStats(prev => ({ ...prev, [key]: target }));
          clearInterval(timer);
        } else {
          setStats(prev => ({ ...prev, [key]: Math.floor(start) }));
        }
      }, duration / steps);
    };

    increment(5200, 'users');
    increment(50000, 'predictions');
    increment(98.5, 'accuracy');
    increment(15, 'papers');
  }, []);

  const GRADIENTS = {
    purple: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    green: 'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)',
    orange: 'linear-gradient(135deg, #ee0979 0%, #ff6a00 100%)',
    blue: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
    pink: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    teal: 'linear-gradient(135deg, #0ba360 0%, #3cba92 100%)'
  };

  const achievements = [
    { icon: <Award size={24} />, text: 'Best AI Innovation 2024', count: '1st', color: '#fbbf24', gradient: GRADIENTS.orange },
    { icon: <CheckCircle size={24} />, text: 'ISO 27001 Certified', count: '99.9%', color: '#10b981', gradient: GRADIENTS.green },
    { icon: <Users size={24} />, text: 'Active Users', count: `${stats.users.toLocaleString()}+`, color: '#3b82f6', gradient: GRADIENTS.blue },
    { icon: <Star size={24} />, text: 'Research Papers', count: `${stats.papers}+`, color: '#8b5cf6', gradient: GRADIENTS.purple }
  ];

  const features = [
    {
      icon: <Cpu size={32} />,
      title: 'Advanced AI Models',
      description: 'State-of-the-art deep learning models for fish classification and disease detection',
      stats: '98.5% accuracy',
      gradient: GRADIENTS.purple,
      color: '#667eea'
    },
    {
      icon: <Eye size={32} />,
      title: 'Computer Vision',
      description: 'Cutting-edge image recognition technology trained on thousands of aquatic species',
      stats: '10,000+ images',
      gradient: GRADIENTS.pink,
      color: '#f093fb'
    },
    {
      icon: <Database size={32} />,
      title: 'Big Data Analytics',
      description: 'Scalable data processing pipeline handling millions of data points',
      stats: '1M+ data points',
      gradient: GRADIENTS.blue,
      color: '#4facfe'
    },
    {
      icon: <Cloud size={32} />,
      title: 'Cloud Infrastructure',
      description: 'Robust AWS-powered architecture ensuring high availability',
      stats: '99.9% uptime',
      gradient: GRADIENTS.green,
      color: '#43e97b'
    },
    {
      icon: <Zap size={32} />,
      title: 'Real-time Processing',
      description: 'Instant AI predictions and analytics with minimal latency',
      stats: '< 2 seconds',
      gradient: GRADIENTS.orange,
      color: '#fa709a'
    },
    {
      icon: <Rocket size={32} />,
      title: 'Continuous Innovation',
      description: 'Regular updates with new features based on latest research',
      stats: 'Monthly updates',
      gradient: GRADIENTS.teal,
      color: '#30cfd0'
    }
  ];

  const techStack = [
    { name: 'React.js', level: 95, color: '#61dafb', icon: '⚛️' },
    { name: 'Node.js', level: 92, color: '#68a063', icon: '🟢' },
    { name: 'Python', level: 98, color: '#3776ab', icon: '🐍' },
    { name: 'TensorFlow', level: 90, color: '#ff6f00', icon: '🧠' },
    { name: 'PyTorch', level: 88, color: '#ee4c2c', icon: '🔥' },
    { name: 'MongoDB', level: 85, color: '#47a248', icon: '🍃' },
    { name: 'PostgreSQL', level: 87, color: '#336791', icon: '🐘' },
    { name: 'AWS', level: 82, color: '#ff9900', icon: '☁️' }
  ];

  const milestones = [
    { year: '2024', title: 'Platform Launch', description: 'MeenaSetu AI officially launched with core features', icon: '🚀', color: '#667eea' },
    { year: '2024', title: '50,000+ Predictions', description: 'Crossed major milestone in AI predictions', icon: '📈', color: '#10b981' },
    { year: '2024', title: '98.5% Accuracy', description: 'Achieved industry-leading accuracy rate', icon: '🎯', color: '#f59e0b' },
    { year: '2025', title: 'Global Expansion', description: 'Serving researchers in 50+ countries', icon: '🌍', color: '#8b5cf6' }
  ];

  const teamMembers = [
    { name: 'Amrish Tiwary', role: 'Founder & Lead Developer', avatar: 'AT', skills: ['AI/ML', 'Full Stack', 'Cloud'], color: '#667eea' },
    { name: 'Dr. Meena Sharma', role: 'Marine Biology Advisor', avatar: 'MS', skills: ['Marine Biology', 'Research', 'Conservation'], color: '#10b981' },
    { name: 'Alex Johnson', role: 'AI Research Lead', avatar: 'AJ', skills: ['Deep Learning', 'CV', 'NLP'], color: '#f59e0b' }
  ];

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(to bottom, #f8fafc 0%, #e2e8f0 100%)',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
      {/* Scroll Progress Bar */}
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        height: 4,
        background: 'rgba(102, 126, 234, 0.1)',
        zIndex: 9999
      }}>
        <div style={{
          height: '100%',
          width: `${scrollProgress}%`,
          background: GRADIENTS.purple,
          transition: 'width 0.1s ease'
        }} />
      </div>

      {/* Floating Action Buttons */}
      {showScrollTop && (
        <button
          onClick={scrollToTop}
          style={{
            position: 'fixed',
            bottom: 32,
            right: 32,
            width: 56,
            height: 56,
            borderRadius: '50%',
            border: 'none',
            background: GRADIENTS.purple,
            color: 'white',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 8px 32px rgba(102, 126, 234, 0.4)',
            zIndex: 1000,
            transition: 'all 0.3s ease',
            animation: 'fadeIn 0.3s ease'
          }}
          onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-4px)'}
          onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}
        >
          <ArrowUp size={24} />
        </button>
      )}

      {/* Hero Section */}
      <div id="hero" style={{
        position: 'relative',
        overflow: 'hidden',
        background: 'linear-gradient(135deg, #0a1929 0%, #1a2332 50%, #0d2b36 100%)',
        color: 'white',
        padding: '8rem 2rem 6rem',
        marginBottom: '4rem'
      }}>
        {/* Animated Background */}
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          opacity: 0.3,
          background: `
            radial-gradient(circle at 20% 80%, rgba(102, 126, 234, 0.2) 0%, transparent 40%),
            radial-gradient(circle at 80% 20%, rgba(67, 233, 123, 0.2) 0%, transparent 40%)
          `
        }} />

        <div style={{ maxWidth: '1200px', margin: '0 auto', position: 'relative', zIndex: 1 }}>
          <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
            <div style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.75rem 1.5rem',
              background: 'rgba(255,255,255,0.1)',
              backdropFilter: 'blur(10px)',
              borderRadius: '30px',
              border: '1px solid rgba(255,255,255,0.2)',
              marginBottom: '2rem',
              animation: 'pulse 2s infinite'
            }}>
              <Zap size={20} />
              <span style={{ fontWeight: '700', fontSize: '1rem' }}>Pioneering AI in Aquatic Research</span>
            </div>

            <h1 style={{
              fontSize: 'clamp(2.5rem, 6vw, 4rem)',
              fontWeight: '900',
              marginBottom: '1.5rem',
              background: 'linear-gradient(135deg, #fff 0%, #4fc3f7 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              lineHeight: 1.2
            }}>
              Revolutionizing Aquatic Intelligence
            </h1>

            <p style={{
              fontSize: 'clamp(1rem, 2vw, 1.25rem)',
              opacity: 0.9,
              maxWidth: '800px',
              margin: '0 auto 3rem',
              lineHeight: 1.6
            }}>
              Powered by cutting-edge artificial intelligence, MeenaSetu transforms how researchers 
              study and protect marine ecosystems. Join the future of aquatic conservation today.
            </p>

            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
              <button
                onClick={() => setVideoOpen(true)}
                style={{
                  padding: '1rem 2rem',
                  borderRadius: '12px',
                  border: 'none',
                  background: GRADIENTS.purple,
                  color: 'white',
                  fontSize: '1.1rem',
                  fontWeight: '700',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  boxShadow: '0 8px 24px rgba(102, 126, 234, 0.4)',
                  transition: 'all 0.3s ease'
                }}
                onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-2px)'}
                onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}
              >
                <PlayCircle size={20} />
                Watch Demo
              </button>

              <button style={{
                padding: '1rem 2rem',
                borderRadius: '12px',
                border: '2px solid rgba(255,255,255,0.3)',
                background: 'transparent',
                color: 'white',
                fontSize: '1.1rem',
                fontWeight: '700',
                cursor: 'pointer',
                transition: 'all 0.3s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(255,255,255,0.1)';
                e.currentTarget.style.transform = 'translateY(-2px)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'transparent';
                e.currentTarget.style.transform = 'translateY(0)';
              }}>
                Get Started
              </button>
            </div>
          </div>

          {/* Live Stats */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
            gap: '1.5rem',
            marginTop: '4rem'
          }}>
            {achievements.map((achievement, index) => (
              <div key={index} style={{
                padding: '2rem',
                background: 'rgba(255,255,255,0.05)',
                backdropFilter: 'blur(20px)',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '16px',
                textAlign: 'center',
                transition: 'all 0.3s ease',
                cursor: 'pointer'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-8px)';
                e.currentTarget.style.background = 'rgba(255,255,255,0.08)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.background = 'rgba(255,255,255,0.05)';
              }}>
                <div style={{
                  fontSize: '2.5rem',
                  fontWeight: '700',
                  color: achievement.color,
                  marginBottom: '0.5rem'
                }}>
                  {achievement.count}
                </div>
                <div style={{ color: achievement.color, marginBottom: '0.5rem' }}>
                  {achievement.icon}
                </div>
                <div style={{ fontSize: '0.9rem', opacity: 0.9, fontWeight: '600' }}>
                  {achievement.text}
                </div>
              </div>
            ))}
          </div>

          {/* Scroll Indicator */}
          <div style={{
            textAlign: 'center',
            marginTop: '4rem',
            animation: 'bounce 2s infinite'
          }}>
            <MousePointer size={32} style={{ opacity: 0.5 }} />
            <div style={{ fontSize: '0.85rem', opacity: 0.5, marginTop: '0.5rem' }}>
              Scroll to explore
            </div>
          </div>
        </div>
      </div>

      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 2rem' }}>
        {/* Mission & Vision */}
        <div id="mission" style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))',
          gap: '2rem',
          marginBottom: '6rem'
        }}>
          <div style={{
            padding: '3rem',
            background: 'white',
            borderRadius: '20px',
            boxShadow: '0 8px 32px rgba(0,0,0,0.08)',
            border: '2px solid rgba(102, 126, 234, 0.1)',
            position: 'relative',
            overflow: 'hidden',
            transition: 'all 0.3s ease'
          }}
          onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-4px)'}
          onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}>
            <div style={{
              position: 'absolute',
              top: -50,
              right: -50,
              width: 150,
              height: 150,
              borderRadius: '50%',
              background: 'radial-gradient(circle, rgba(102, 126, 234, 0.1) 0%, transparent 70%)'
            }} />
            
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.5rem', position: 'relative', zIndex: 1 }}>
              <div style={{
                width: 56,
                height: 56,
                borderRadius: '12px',
                background: GRADIENTS.purple,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white'
              }}>
                <Target size={28} />
              </div>
              <h2 style={{ margin: 0, fontSize: '2rem', fontWeight: '800' }}>Our Mission</h2>
            </div>

            <p style={{ fontSize: '1.1rem', lineHeight: 1.8, color: '#475569', position: 'relative', zIndex: 1 }}>
              To democratize access to advanced AI technology for aquatic research, 
              making world-class fish species identification and disease detection 
              accessible to researchers, conservationists, and aquaculture professionals 
              worldwide. We're committed to protecting marine biodiversity through 
              innovative technology.
            </p>
          </div>

          <div style={{
            padding: '3rem',
            background: 'white',
            borderRadius: '20px',
            boxShadow: '0 8px 32px rgba(0,0,0,0.08)',
            border: '2px solid rgba(240, 147, 251, 0.1)',
            position: 'relative',
            overflow: 'hidden',
            transition: 'all 0.3s ease'
          }}
          onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-4px)'}
          onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}>
            <div style={{
              position: 'absolute',
              top: -50,
              right: -50,
              width: 150,
              height: 150,
              borderRadius: '50%',
              background: 'radial-gradient(circle, rgba(240, 147, 251, 0.1) 0%, transparent 70%)'
            }} />
            
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.5rem', position: 'relative', zIndex: 1 }}>
              <div style={{
                width: 56,
                height: 56,
                borderRadius: '12px',
                background: GRADIENTS.pink,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white'
              }}>
                <Lightbulb size={28} />
              </div>
              <h2 style={{ margin: 0, fontSize: '2rem', fontWeight: '800' }}>Our Vision</h2>
            </div>

            <p style={{ fontSize: '1.1rem', lineHeight: 1.8, color: '#475569', position: 'relative', zIndex: 1 }}>
              To become the world's leading AI platform for aquatic intelligence, 
              enabling groundbreaking discoveries in marine biology, sustainable 
              aquaculture, and ocean conservation. We envision a future where AI-powered 
              insights help preserve our oceans for generations to come.
            </p>
          </div>
        </div>

        {/* Platform Features */}
        <div id="features" style={{ marginBottom: '6rem' }}>
          <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
            <h2 style={{
              fontSize: 'clamp(2rem, 4vw, 3rem)',
              fontWeight: '900',
              marginBottom: '1rem',
              background: GRADIENTS.purple,
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent'
            }}>
              Advanced Platform Features
            </h2>
            <p style={{ fontSize: '1.2rem', color: '#64748b', maxWidth: '800px', margin: '0 auto' }}>
              Built with cutting-edge technology to deliver exceptional performance, accuracy, and user experience
            </p>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))',
            gap: '2rem'
          }}>
            {features.map((feature, index) => (
              <div key={index} style={{
                padding: '2.5rem',
                background: 'white',
                borderRadius: '20px',
                boxShadow: '0 8px 32px rgba(0,0,0,0.08)',
                transition: 'all 0.3s ease',
                cursor: 'pointer',
                position: 'relative',
                overflow: 'hidden'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-8px)';
                e.currentTarget.style.boxShadow = '0 16px 48px rgba(0,0,0,0.12)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 8px 32px rgba(0,0,0,0.08)';
              }}>
                <div style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  height: 4,
                  background: feature.gradient
                }} />

                <div style={{
                  width: 80,
                  height: 80,
                  borderRadius: '16px',
                  background: feature.gradient,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                  margin: '0 auto 1.5rem',
                  boxShadow: `0 8px 24px ${feature.color}40`
                }}>
                  {feature.icon}
                </div>

                <h3 style={{ fontSize: '1.5rem', fontWeight: '800', marginBottom: '1rem', textAlign: 'center' }}>
                  {feature.title}
                </h3>

                <p style={{ fontSize: '1rem', lineHeight: 1.7, color: '#64748b', marginBottom: '1.5rem', textAlign: 'center' }}>
                  {feature.description}
                </p>

                <div style={{
                  display: 'inline-block',
                  padding: '0.5rem 1rem',
                  background: feature.gradient,
                  color: 'white',
                  borderRadius: '20px',
                  fontSize: '0.9rem',
                  fontWeight: '700',
                  margin: '0 auto',
                  display: 'block',
                  width: 'fit-content',
                  textAlign: 'center'
                }}>
                  {feature.stats}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Creator Section */}
        <div id="creator" style={{
          padding: '4rem',
          background: 'linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%)',
          borderRadius: '24px',
          color: 'white',
          marginBottom: '6rem',
          boxShadow: '0 20px 60px rgba(15, 32, 39, 0.4)',
          position: 'relative',
          overflow: 'hidden'
        }}>
          <div style={{
            position: 'absolute',
            top: -100,
            right: -100,
            width: 300,
            height: 300,
            borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(102, 126, 234, 0.2) 0%, transparent 70%)',
            animation: 'rotate 20s linear infinite'
          }} />

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
            gap: '3rem',
            alignItems: 'center',
            position: 'relative',
            zIndex: 1
          }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{
                width: 200,
                height: 200,
                borderRadius: '50%',
                background: GRADIENTS.purple,
                margin: '0 auto 2rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '4rem',
                fontWeight: '800',
                border: '4px solid rgba(255,255,255,0.2)',
                boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
                transition: 'all 0.3s ease',
                cursor: 'pointer'
              }}
              onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.05)'}
              onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}>
                AT
              </div>

              <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', marginBottom: '1.5rem' }}>
                {[
                  { icon: <Linkedin size={20} />, color: '#0077b5' },
                  { icon: <Github size={20} />, color: '#333' },
                  { icon: <Mail size={20} />, color: '#ea4335' }
                ].map((social, idx) => (
                  <div key={idx} style={{
                    width: 48,
                    height: 48,
                    borderRadius: '50%',
                    background: social.color,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.2) rotate(5deg)'}
                  onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1) rotate(0deg)'}>
                    {social.icon}
                  </div>
                ))}
              </div>

              <div style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.5rem 1rem',
                background: 'rgba(16, 185, 129, 0.2)',
                border: '1px solid rgba(16, 185, 129, 0.3)',
                borderRadius: '20px',
                fontSize: '0.85rem',
                fontWeight: '600',
                color: '#10b981'
              }}>
                <CheckCircle size={16} />
                Verified Profile
              </div>
            </div>

            <div>
              <div style={{
                display: 'inline-block',
                padding: '0.5rem 1rem',
                background: 'rgba(255,255,255,0.1)',
                borderRadius: '20px',
                fontSize: '0.9rem',
                fontWeight: '700',
                marginBottom: '1rem'
              }}>
                Creator & Lead Developer
              </div>

              <h2 style={{ fontSize: '2.5rem', fontWeight: '900', marginBottom: '0.5rem' }}>
                Amrish Kumar Tiwary
              </h2>

              <h3 style={{ fontSize: '1.3rem', color: '#4fc3f7', marginBottom: '2rem', fontWeight: '600' }}>
                Full Stack AI Engineer | Machine Learning Expert | Visionary
              </h3>

              <p style={{ fontSize: '1rem', lineHeight: 1.8, opacity: 0.95, marginBottom: '1.5rem' }}>
                A passionate Full Stack AI Engineer with expertise in building intelligent systems 
                that solve real-world environmental challenges. With extensive experience in machine learning, 
                computer vision, and full-stack development, Amrish created MeenaSetu AI to bridge 
                the critical gap between cutting-edge AI technology and aquatic research needs.
              </p>

              <p style={{ fontSize: '1rem', lineHeight: 1.8, opacity: 0.95, marginBottom: '2rem' }}>
                Specializing in Python, TensorFlow, PyTorch, React, and cloud technologies, 
                he leads a team dedicated to leveraging AI for environmental conservation and sustainable 
                development.
              </p>

              <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                {[
                  { skill: 'AI/ML', level: 98, color: '#667eea' },
                  { skill: 'Deep Learning', level: 96, color: '#10b981' },
                  { skill: 'Full Stack', level: 95, color: '#f59e0b' },
                  { skill: 'Cloud', level: 92, color: '#8b5cf6' }
                ].map((item, idx) => (
                  <div key={idx} style={{
                    padding: '0.5rem 1rem',
                    background: `${item.color}20`,
                    border: `1px solid ${item.color}40`,
                    borderRadius: '20px',
                    fontSize: '0.9rem',
                    fontWeight: '700',
                    color: item.color,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    transition: 'all 0.3s ease',
                    cursor: 'pointer'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.05)'}
                  onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}>
                    <TrendingUp size={14} />
                    {item.skill} {item.level}%
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Tech Stack */}
        <div id="tech" style={{ marginBottom: '6rem' }}>
          <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
            <h2 style={{
              fontSize: 'clamp(2rem, 4vw, 3rem)',
              fontWeight: '900',
              marginBottom: '1rem',
              background: GRADIENTS.purple,
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent'
            }}>
              Technology Stack
            </h2>
            <p style={{ fontSize: '1.2rem', color: '#64748b' }}>
              Built with industry-leading technologies and frameworks
            </p>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
            gap: '2rem'
          }}>
            {techStack.map((tech, index) => (
              <div key={index} style={{
                padding: '2rem',
                background: 'white',
                borderRadius: '16px',
                boxShadow: '0 4px 16px rgba(0,0,0,0.08)',
                transition: 'all 0.3s ease',
                cursor: 'pointer'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-4px)';
                e.currentTarget.style.boxShadow = '0 8px 24px rgba(0,0,0,0.12)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 4px 16px rgba(0,0,0,0.08)';
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
                  <span style={{ fontSize: '2rem' }}>{tech.icon}</span>
                  <div style={{ flex: 1 }}>
                    <h4 style={{ margin: 0, fontSize: '1.1rem', fontWeight: '700' }}>{tech.name}</h4>
                    <div style={{ fontSize: '0.9rem', fontWeight: '800', color: tech.color }}>
                      {tech.level}%
                    </div>
                  </div>
                </div>

                <div style={{
                  height: 8,
                  background: `${tech.color}20`,
                  borderRadius: '4px',
                  overflow: 'hidden'
                }}>
                  <div style={{
                    width: `${tech.level}%`,
                    height: '100%',
                    background: tech.color,
                    borderRadius: '4px',
                    transition: 'width 1s ease',
                    animation: 'progressBar 1.5s ease'
                  }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Journey Timeline */}
        <div id="journey" style={{ marginBottom: '6rem' }}>
          <h2 style={{
            fontSize: 'clamp(2rem, 4vw, 3rem)',
            fontWeight: '900',
            textAlign: 'center',
            marginBottom: '4rem',
            background: GRADIENTS.purple,
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent'
          }}>
            Our Journey Timeline
          </h2>

          <div style={{ position: 'relative' }}>
            {milestones.map((milestone, index) => (
              <div key={index} style={{
                position: 'relative',
                paddingLeft: 'clamp(4rem, 8vw, 6rem)',
                marginBottom: '3rem'
              }}>
                <div style={{
                  position: 'absolute',
                  left: '2rem',
                  top: '1.5rem',
                  width: 20,
                  height: 20,
                  borderRadius: '50%',
                  background: milestone.color,
                  border: '4px solid white',
                  boxShadow: `0 0 0 4px ${milestone.color}40`,
                  zIndex: 2
                }} />

                {index < milestones.length - 1 && (
                  <div style={{
                    position: 'absolute',
                    left: '2.6rem',
                    top: '3rem',
                    width: 4,
                    height: 'calc(100% + 1rem)',
                    background: `linear-gradient(180deg, ${milestone.color} 0%, ${milestones[index + 1].color} 100%)`
                  }} />
                )}

                <div style={{
                  padding: '2rem',
                  background: 'white',
                  borderRadius: '16px',
                  boxShadow: '0 4px 16px rgba(0,0,0,0.08)',
                  transition: 'all 0.3s ease',
                  cursor: 'pointer'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateX(10px)';
                  e.currentTarget.style.boxShadow = '0 8px 24px rgba(0,0,0,0.12)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateX(0)';
                  e.currentTarget.style.boxShadow = '0 4px 16px rgba(0,0,0,0.08)';
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
                    <div style={{
                      width: 48,
                      height: 48,
                      borderRadius: '12px',
                      background: `linear-gradient(135deg, ${milestone.color} 0%, ${milestone.color}cc 100%)`,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '1.5rem'
                    }}>
                      {milestone.icon}
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{
                        display: 'inline-block',
                        padding: '0.25rem 0.75rem',
                        background: `${milestone.color}20`,
                        color: milestone.color,
                        borderRadius: '12px',
                        fontSize: '0.85rem',
                        fontWeight: '800',
                        marginBottom: '0.5rem'
                      }}>
                        {milestone.year}
                      </div>
                      <h3 style={{ margin: 0, fontSize: '1.3rem', fontWeight: '800' }}>
                        {milestone.title}
                      </h3>
                    </div>
                  </div>
                  <p style={{ fontSize: '1rem', color: '#64748b', margin: 0 }}>
                    {milestone.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Team Members */}
        <div id="team" style={{ marginBottom: '6rem' }}>
          <h2 style={{
            fontSize: 'clamp(2rem, 4vw, 3rem)',
            fontWeight: '900',
            textAlign: 'center',
            marginBottom: '4rem',
            background: GRADIENTS.purple,
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent'
          }}>
            Meet Our Team
          </h2>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
            gap: '2rem'
          }}>
            {teamMembers.map((member, index) => (
              <div key={index} style={{
                padding: '2.5rem',
                background: 'white',
                borderRadius: '20px',
                boxShadow: '0 8px 32px rgba(0,0,0,0.08)',
                textAlign: 'center',
                transition: 'all 0.3s ease',
                cursor: 'pointer',
                position: 'relative',
                overflow: 'hidden'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-8px)';
                e.currentTarget.style.boxShadow = '0 16px 48px rgba(0,0,0,0.12)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 8px 32px rgba(0,0,0,0.08)';
              }}>
                <div style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  height: 4,
                  background: `linear-gradient(90deg, ${member.color} 0%, ${member.color}cc 100%)`
                }} />

                <div style={{
                  width: 100,
                  height: 100,
                  borderRadius: '50%',
                  background: `linear-gradient(135deg, ${member.color} 0%, ${member.color}cc 100%)`,
                  margin: '0 auto 1.5rem',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '2.5rem',
                  fontWeight: '800',
                  color: 'white',
                  border: `4px solid ${member.color}40`
                }}>
                  {member.avatar}
                </div>

                <h3 style={{ fontSize: '1.5rem', fontWeight: '800', marginBottom: '0.5rem' }}>
                  {member.name}
                </h3>

                <p style={{ fontSize: '1rem', color: member.color, fontWeight: '600', marginBottom: '1.5rem' }}>
                  {member.role}
                </p>

                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', justifyContent: 'center' }}>
                  {member.skills.map((skill, idx) => (
                    <span key={idx} style={{
                      padding: '0.4rem 0.8rem',
                      background: `${member.color}20`,
                      border: `1px solid ${member.color}40`,
                      borderRadius: '12px',
                      fontSize: '0.85rem',
                      fontWeight: '600',
                      color: member.color
                    }}>
                      {skill}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Final CTA */}
        <div style={{
          padding: 'clamp(3rem, 8vw, 5rem)',
          background: 'white',
          borderRadius: '24px',
          boxShadow: '0 20px 60px rgba(102, 126, 234, 0.15)',
          textAlign: 'center',
          position: 'relative',
          overflow: 'hidden',
          marginBottom: '4rem'
        }}>
          <div style={{
            position: 'absolute',
            top: -100,
            right: -100,
            width: 300,
            height: 300,
            borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(102, 126, 234, 0.1) 0%, transparent 70%)'
          }} />
          
          <div style={{
            position: 'absolute',
            bottom: -100,
            left: -100,
            width: 300,
            height: 300,
            borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(118, 75, 162, 0.1) 0%, transparent 70%)'
          }} />

          <div style={{ position: 'relative', zIndex: 1 }}>
            <h2 style={{ fontSize: 'clamp(2rem, 4vw, 3rem)', fontWeight: '900', marginBottom: '1rem' }}>
              Join the AI Revolution in Aquatic Research
            </h2>

            <p style={{ fontSize: '1.2rem', color: '#64748b', marginBottom: '3rem', maxWidth: '800px', margin: '0 auto 3rem' }}>
              Be part of a global community using cutting-edge AI to protect our oceans and advance marine science
            </p>

            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap', marginBottom: '3rem' }}>
              <button style={{
                padding: '1.25rem 2.5rem',
                borderRadius: '12px',
                border: 'none',
                background: GRADIENTS.purple,
                color: 'white',
                fontSize: '1.1rem',
                fontWeight: '800',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                boxShadow: '0 8px 24px rgba(102, 126, 234, 0.4)',
                transition: 'all 0.3s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-4px)';
                e.currentTarget.style.boxShadow = '0 16px 40px rgba(102, 126, 234, 0.5)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 8px 24px rgba(102, 126, 234, 0.4)';
              }}>
                <Rocket size={20} />
                Start Free Trial
              </button>

              <button style={{
                padding: '1.25rem 2.5rem',
                borderRadius: '12px',
                border: '2px solid #667eea',
                background: 'transparent',
                color: '#667eea',
                fontSize: '1.1rem',
                fontWeight: '800',
                cursor: 'pointer',
                transition: 'all 0.3s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(102, 126, 234, 0.1)';
                e.currentTarget.style.transform = 'translateY(-4px)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'transparent';
                e.currentTarget.style.transform = 'translateY(0)';
              }}>
                Schedule Demo
              </button>
            </div>

            <div style={{ display: 'flex', gap: '2rem', justifyContent: 'center', flexWrap: 'wrap' }}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.75rem 1.5rem',
                background: 'rgba(102, 126, 234, 0.1)',
                borderRadius: '25px',
                fontSize: '1rem',
                fontWeight: '700',
                color: '#667eea'
              }}>
                <Activity size={20} />
                {stats.users.toLocaleString()}+ Active Researchers
              </div>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.75rem 1.5rem',
                background: 'rgba(67, 233, 123, 0.1)',
                borderRadius: '25px',
                fontSize: '1rem',
                fontWeight: '700',
                color: '#43e97b'
              }}>
                <Shield size={20} />
                {stats.predictions.toLocaleString()}+ Predictions Made
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Video Dialog */}
      {videoOpen && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0,0,0,0.8)',
          backdropFilter: 'blur(8px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999,
          padding: '2rem',
          animation: 'fadeIn 0.3s ease'
        }}
        onClick={() => setVideoOpen(false)}>
          <div style={{
            maxWidth: '900px',
            width: '100%',
            background: 'linear-gradient(135deg, #0a1929 0%, #0d2b36 100%)',
            borderRadius: '20px',
            overflow: 'hidden',
            position: 'relative',
            animation: 'slideUp 0.3s ease'
          }}
          onClick={(e) => e.stopPropagation()}>
            <button
              onClick={() => setVideoOpen(false)}
              style={{
                position: 'absolute',
                top: 16,
                right: 16,
                width: 40,
                height: 40,
                borderRadius: '50%',
                border: 'none',
                background: 'rgba(0,0,0,0.5)',
                color: 'white',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 10,
                transition: 'all 0.3s ease'
              }}
              onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(0,0,0,0.8)'}
              onMouseLeave={(e) => e.currentTarget.style.background = 'rgba(0,0,0,0.5)'}
            >
              <X size={20} />
            </button>

            <div style={{
              aspectRatio: '16/9',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'white',
              fontSize: '1.5rem',
              fontWeight: '700'
            }}>
              Demo Video Placeholder
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes pulse {
          0%, 100% { box-shadow: 0 0 0 0 rgba(255,255,255,0.4); }
          50% { box-shadow: 0 0 0 10px rgba(255,255,255,0); }
        }
        @keyframes bounce {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-10px); }
        }
        @keyframes rotate {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(30px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes progressBar {
          from { width: 0; }
        }
        ::-webkit-scrollbar {
          width: 10px;
        }
        ::-webkit-scrollbar-track {
          background: #f1f5f9;
        }
        ::-webkit-scrollbar-thumb {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          border-radius: 10px;
        }
      `}</style>
    </div>
  );
};

export default About;