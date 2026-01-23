import React, { useState, useEffect, useCallback } from 'react';
import { 
  Mail, Phone, MapPin, Send, Facebook, Twitter, Linkedin, Instagram, 
  Clock, CheckCircle, AlertCircle, Map, Building, Users, Globe,
  MessageSquare, Zap, Award, Heart, Star, ArrowRight, Sparkles
} from 'lucide-react';

const Contact = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });

  const [errors, setErrors] = useState({});
  const [touched, setTouched] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitSuccess, setSubmitSuccess] = useState(false);
  const [visible, setVisible] = useState(false);
  const [hoveredCard, setHoveredCard] = useState(null);
  const [hoveredSocial, setHoveredSocial] = useState(null);
  const [focusedInput, setFocusedInput] = useState(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [particlesVisible, setParticlesVisible] = useState(false);

  useEffect(() => {
    setVisible(true);
    setTimeout(() => setParticlesVisible(true), 500);
  }, []);

  useEffect(() => {
    const handleMouseMove = (e) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const theme = {
    gradients: {
      purple: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      green: 'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)',
      orange: 'linear-gradient(135deg, #ee0979 0%, #ff6a00 100%)',
      blue: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
      pink: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
      gold: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)'
    },
    colors: {
      primary: '#667eea',
      secondary: '#764ba2',
      success: '#10b981',
      error: '#ef4444',
      warning: '#f59e0b',
      text: '#1e293b',
      textSecondary: '#64748b',
      border: '#e2e8f0'
    }
  };

  const validateField = useCallback((name, value) => {
    const trimmedValue = value.trim();
    
    switch (name) {
      case 'name':
        if (!trimmedValue) return 'Name is required';
        return trimmedValue.length < 2 ? 'Name must be at least 2 characters' : '';
      case 'email':
        if (!trimmedValue) return 'Email is required';
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return !emailRegex.test(trimmedValue) ? 'Please enter a valid email address' : '';
      case 'subject':
        if (!trimmedValue) return 'Subject is required';
        return trimmedValue.length < 3 ? 'Subject must be at least 3 characters' : '';
      case 'message':
        if (!trimmedValue) return 'Message is required';
        return trimmedValue.length < 10 ? 'Message must be at least 10 characters' : '';
      default:
        return '';
    }
  }, []);

  const handleChange = useCallback((e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    
    if (touched[name]) {
      setErrors(prev => ({ ...prev, [name]: validateField(name, value) }));
    }
  }, [touched, validateField]);

  const handleBlur = useCallback((e) => {
    const { name, value } = e.target;
    setTouched(prev => ({ ...prev, [name]: true }));
    setErrors(prev => ({ ...prev, [name]: validateField(name, value) }));
  }, [validateField]);

  const handleSubmit = useCallback(async (e) => {
    e.preventDefault();
    
    const newTouched = {};
    const newErrors = {};
    let hasErrors = false;

    Object.keys(formData).forEach(key => {
      newTouched[key] = true;
      const error = validateField(key, formData[key]);
      if (error) {
        newErrors[key] = error;
        hasErrors = true;
      }
    });

    setTouched(newTouched);
    setErrors(newErrors);

    if (!hasErrors) {
      setIsSubmitting(true);
      
      try {
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        setIsSubmitting(false);
        setSubmitSuccess(true);
        setFormData({ name: '', email: '', subject: '', message: '' });
        setTouched({});
        
        setTimeout(() => {
          setSubmitSuccess(false);
        }, 5000);
      } catch (error) {
        setIsSubmitting(false);
      }
    }
  }, [formData, validateField]);

  const contactInfo = [
    {
      icon: <Phone size={28} />,
      title: 'Phone',
      primary: '+91 1234 567 890',
      secondary: 'Mon-Fri 9am-6pm IST',
      link: 'tel:+911234567890',
      gradient: theme.gradients.blue,
      color: '#4facfe'
    },
    {
      icon: <Mail size={28} />,
      title: 'Email',
      primary: 'contact@meenasetu.com',
      secondary: 'We reply within 24 hours',
      link: 'mailto:contact@meenasetu.com',
      gradient: theme.gradients.purple,
      color: '#667eea'
    },
    {
      icon: <MapPin size={28} />,
      title: 'Address',
      primary: 'Fraser Road Area',
      secondary: 'Patna, Bihar 800001, India',
      link: '#map',
      gradient: theme.gradients.green,
      color: '#11998e'
    },
    {
      icon: <Clock size={28} />,
      title: 'Business Hours',
      primary: 'Mon-Fri: 9:00 AM - 6:00 PM',
      secondary: 'Saturday: 10:00 AM - 4:00 PM',
      link: '#',
      gradient: theme.gradients.orange,
      color: '#ee0979'
    }
  ];

  const socialMedia = [
    { icon: <Facebook size={20} />, name: 'Facebook', link: '#', color: '#1877f2' },
    { icon: <Twitter size={20} />, name: 'Twitter', link: '#', color: '#1da1f2' },
    { icon: <Linkedin size={20} />, name: 'LinkedIn', link: '#', color: '#0a66c2' },
    { icon: <Instagram size={20} />, name: 'Instagram', link: '#', color: '#e4405f' }
  ];

  const stats = [
    { icon: <MessageSquare />, value: '2.5k+', label: 'Messages Sent', gradient: theme.gradients.purple },
    { icon: <Users />, value: '500+', label: 'Happy Clients', gradient: theme.gradients.green },
    { icon: <Zap />, value: '24/7', label: 'Support Available', gradient: theme.gradients.orange },
    { icon: <Award />, value: '5+', label: 'Years Experience', gradient: theme.gradients.blue }
  ];

  const features = [
    { icon: <CheckCircle />, text: 'Fast Response Time', color: '#10b981' },
    { icon: <Star />, text: 'Expert Support Team', color: '#f59e0b' },
    { icon: <Heart />, text: 'Customer Satisfaction', color: '#ef4444' },
    { icon: <Globe />, text: 'Global Reach', color: '#3b82f6' }
  ];

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(to bottom, #0f172a 0%, #1e293b 50%, #334155 100%)',
      position: 'relative',
      overflow: 'hidden'
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
        
        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-20px); }
        }
        
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.8; transform: scale(1.05); }
        }
        
        @keyframes slideIn {
          from { opacity: 0; transform: translateX(100px); }
          to { opacity: 1; transform: translateX(0); }
        }

        @keyframes shimmer {
          0% { background-position: -1000px 0; }
          100% { background-position: 1000px 0; }
        }

        @keyframes glow {
          0%, 100% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.5); }
          50% { box-shadow: 0 0 40px rgba(102, 126, 234, 0.8); }
        }

        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        .gradient-text {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }

        .glass-effect {
          background: rgba(255, 255, 255, 0.05);
          backdrop-filter: blur(20px);
          border: 1px solid rgba(255, 255, 255, 0.1);
        }

        * {
          box-sizing: border-box;
          margin: 0;
          padding: 0;
        }

        body {
          font-family: 'Inter', sans-serif;
        }
      `}</style>

      {/* Animated Background Particles */}
      {particlesVisible && [...Array(20)].map((_, i) => (
        <div
          key={i}
          style={{
            position: 'absolute',
            width: Math.random() * 4 + 2 + 'px',
            height: Math.random() * 4 + 2 + 'px',
            borderRadius: '50%',
            background: `rgba(${102 + Math.random() * 100}, ${126 + Math.random() * 100}, 234, ${Math.random() * 0.5 + 0.3})`,
            top: Math.random() * 100 + '%',
            left: Math.random() * 100 + '%',
            animation: `float ${Math.random() * 10 + 10}s ease-in-out infinite`,
            animationDelay: `${Math.random() * 5}s`,
            pointerEvents: 'none'
          }}
        />
      ))}

      {/* Success Notification */}
      {submitSuccess && (
        <div style={{
          position: 'fixed',
          top: '30px',
          right: '30px',
          background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
          color: 'white',
          padding: '20px 28px',
          borderRadius: '16px',
          boxShadow: '0 20px 60px rgba(16, 185, 129, 0.4)',
          display: 'flex',
          alignItems: 'center',
          gap: '14px',
          zIndex: 1000,
          animation: 'slideIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)',
          backdropFilter: 'blur(10px)'
        }}>
          <CheckCircle size={28} />
          <div>
            <div style={{ fontWeight: '700', marginBottom: '4px', fontSize: '1.1rem' }}>Success!</div>
            <div style={{ fontSize: '0.9rem', opacity: 0.95 }}>
              Message sent! We'll respond within 24 hours.
            </div>
          </div>
        </div>
      )}

      {/* Hero Section */}
      <div style={{
        padding: '100px 20px 80px',
        textAlign: 'center',
        position: 'relative',
        zIndex: 1
      }}>
        <div style={{
          maxWidth: '900px',
          margin: '0 auto',
          opacity: visible ? 1 : 0,
          transform: visible ? 'translateY(0)' : 'translateY(-30px)',
          transition: 'all 1s cubic-bezier(0.34, 1.56, 0.64, 1)'
        }}>
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '10px',
            padding: '10px 24px',
            background: 'rgba(102, 126, 234, 0.15)',
            backdropFilter: 'blur(10px)',
            borderRadius: '50px',
            border: '1px solid rgba(102, 126, 234, 0.3)',
            marginBottom: '30px',
            animation: 'pulse 3s ease-in-out infinite'
          }}>
            <Sparkles size={18} color="#667eea" />
            <span style={{ color: '#a5b4fc', fontWeight: '600', fontSize: '0.9rem' }}>
              Connect with MeenaSetu AI
            </span>
          </div>

          <h1 style={{
            fontSize: 'clamp(2.5rem, 6vw, 4.5rem)',
            fontWeight: '900',
            color: 'white',
            marginBottom: '24px',
            lineHeight: 1.1,
            letterSpacing: '-2px'
          }}>
            Let's Start a{' '}
            <span className="gradient-text" style={{
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent'
            }}>
              Conversation
            </span>
          </h1>
          
          <p style={{
            fontSize: 'clamp(1.1rem, 2vw, 1.35rem)',
            color: '#94a3b8',
            maxWidth: '700px',
            margin: '0 auto 40px',
            lineHeight: 1.7,
            fontWeight: '400'
          }}>
            Have questions about fish health, species identification, or our AI technology? 
            Our expert team is here to help you 24/7.
          </p>

          {/* Features Grid */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
            gap: '16px',
            maxWidth: '700px',
            margin: '0 auto'
          }}>
            {features.map((feature, idx) => (
              <div key={idx} style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                padding: '12px 16px',
                background: 'rgba(255, 255, 255, 0.05)',
                backdropFilter: 'blur(10px)',
                borderRadius: '12px',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                transition: 'all 0.3s ease',
                cursor: 'pointer'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                e.currentTarget.style.transform = 'translateY(-4px)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                e.currentTarget.style.transform = 'translateY(0)';
              }}>
                <div style={{ color: feature.color }}>
                  {React.cloneElement(feature.icon, { size: 18 })}
                </div>
                <span style={{ color: '#e2e8f0', fontSize: '0.85rem', fontWeight: '600' }}>
                  {feature.text}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div style={{
        maxWidth: '1400px',
        margin: '0 auto',
        padding: '0 20px 80px',
        position: 'relative',
        zIndex: 1
      }}>
        {/* Contact Info Cards */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '24px',
          marginBottom: '60px'
        }}>
          {contactInfo.map((info, index) => (
            <a
              key={index}
              href={info.link}
              style={{
                textDecoration: 'none',
                background: 'rgba(255, 255, 255, 0.05)',
                backdropFilter: 'blur(20px)',
                borderRadius: '24px',
                padding: '32px',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                position: 'relative',
                overflow: 'hidden',
                transition: 'all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)',
                opacity: visible ? 1 : 0,
                transform: visible ? 'translateY(0)' : 'translateY(30px)',
                transitionDelay: `${index * 0.1}s`
              }}
              onMouseEnter={(e) => {
                setHoveredCard(index);
                e.currentTarget.style.transform = 'translateY(-12px) scale(1.02)';
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
              }}
              onMouseLeave={(e) => {
                setHoveredCard(null);
                e.currentTarget.style.transform = 'translateY(0) scale(1)';
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
              }}
            >
              {/* Gradient Overlay */}
              <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                background: info.gradient,
                opacity: hoveredCard === index ? 0.15 : 0,
                transition: 'opacity 0.4s ease',
                pointerEvents: 'none'
              }} />

              <div style={{ position: 'relative', zIndex: 1 }}>
                <div style={{
                  width: '64px',
                  height: '64px',
                  borderRadius: '18px',
                  background: info.gradient,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginBottom: '20px',
                  color: 'white',
                  transition: 'all 0.3s ease',
                  transform: hoveredCard === index ? 'scale(1.1) rotate(5deg)' : 'scale(1)',
                  boxShadow: hoveredCard === index ? `0 20px 40px ${info.color}40` : 'none'
                }}>
                  {info.icon}
                </div>
                
                <div style={{
                  fontSize: '0.75rem',
                  fontWeight: '700',
                  color: '#94a3b8',
                  textTransform: 'uppercase',
                  letterSpacing: '1.5px',
                  marginBottom: '12px'
                }}>
                  {info.title}
                </div>
                
                <div style={{
                  fontSize: '1.2rem',
                  fontWeight: '700',
                  color: 'white',
                  marginBottom: '8px',
                  lineHeight: 1.3
                }}>
                  {info.primary}
                </div>
                
                <div style={{
                  fontSize: '0.9rem',
                  color: '#cbd5e1',
                  lineHeight: 1.5
                }}>
                  {info.secondary}
                </div>
              </div>
            </a>
          ))}
        </div>

        {/* Form and Map Section */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1.2fr 1fr',
          gap: '40px',
          alignItems: 'start'
        }}>
          {/* Contact Form */}
          <div style={{
            background: 'rgba(255, 255, 255, 0.05)',
            backdropFilter: 'blur(20px)',
            borderRadius: '32px',
            padding: '48px',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
            opacity: visible ? 1 : 0,
            transform: visible ? 'translateX(0)' : 'translateX(-30px)',
            transition: 'all 0.8s cubic-bezier(0.34, 1.56, 0.64, 1) 0.3s'
          }}>
            <div style={{ marginBottom: '40px' }}>
              <h2 style={{
                fontSize: 'clamp(1.8rem, 3vw, 2.5rem)',
                fontWeight: '800',
                color: 'white',
                marginBottom: '12px',
                lineHeight: 1.2
              }}>
                Send Us a Message
              </h2>
              <p style={{
                fontSize: '1.05rem',
                color: '#94a3b8',
                lineHeight: 1.6
              }}>
                Fill out the form below and our team will get back to you within 24 hours.
              </p>
            </div>

            <form onSubmit={handleSubmit}>
              {/* Name Input */}
              <div style={{ marginBottom: '24px' }}>
                <label style={{
                  display: 'block',
                  fontSize: '0.9rem',
                  fontWeight: '600',
                  color: '#e2e8f0',
                  marginBottom: '10px',
                  letterSpacing: '0.3px'
                }}>
                  Full Name *
                </label>
                <input
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  onBlur={handleBlur}
                  onFocus={() => setFocusedInput('name')}
                  placeholder="John Doe"
                  style={{
                    width: '100%',
                    padding: '16px 20px',
                    fontSize: '1rem',
                    background: 'rgba(255, 255, 255, 0.05)',
                    border: `2px solid ${errors.name && touched.name ? theme.colors.error : focusedInput === 'name' ? theme.colors.primary : 'rgba(255, 255, 255, 0.1)'}`,
                    borderRadius: '14px',
                    outline: 'none',
                    transition: 'all 0.3s ease',
                    color: 'white',
                    boxShadow: focusedInput === 'name' ? `0 0 0 4px ${theme.colors.primary}20` : 'none'
                  }}
                />
                {errors.name && touched.name && (
                  <div style={{
                    color: theme.colors.error,
                    fontSize: '0.85rem',
                    marginTop: '8px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px'
                  }}>
                    <AlertCircle size={14} />
                    {errors.name}
                  </div>
                )}
              </div>

              {/* Email Input */}
              <div style={{ marginBottom: '24px' }}>
                <label style={{
                  display: 'block',
                  fontSize: '0.9rem',
                  fontWeight: '600',
                  color: '#e2e8f0',
                  marginBottom: '10px',
                  letterSpacing: '0.3px'
                }}>
                  Email Address *
                </label>
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  onBlur={handleBlur}
                  onFocus={() => setFocusedInput('email')}
                  placeholder="john@example.com"
                  style={{
                    width: '100%',
                    padding: '16px 20px',
                    fontSize: '1rem',
                    background: 'rgba(255, 255, 255, 0.05)',
                    border: `2px solid ${errors.email && touched.email ? theme.colors.error : focusedInput === 'email' ? theme.colors.primary : 'rgba(255, 255, 255, 0.1)'}`,
                    borderRadius: '14px',
                    outline: 'none',
                    transition: 'all 0.3s ease',
                    color: 'white',
                    boxShadow: focusedInput === 'email' ? `0 0 0 4px ${theme.colors.primary}20` : 'none'
                  }}
                />
                {errors.email && touched.email && (
                  <div style={{
                    color: theme.colors.error,
                    fontSize: '0.85rem',
                    marginTop: '8px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px'
                  }}>
                    <AlertCircle size={14} />
                    {errors.email}
                  </div>
                )}
              </div>

              {/* Subject Input */}
              <div style={{ marginBottom: '24px' }}>
                <label style={{
                  display: 'block',
                  fontSize: '0.9rem',
                  fontWeight: '600',
                  color: '#e2e8f0',
                  marginBottom: '10px',
                  letterSpacing: '0.3px'
                }}>
                  Subject *
                </label>
                <input
                  type="text"
                  name="subject"
                  value={formData.subject}
                  onChange={handleChange}
                  onBlur={handleBlur}
                  onFocus={() => setFocusedInput('subject')}
                  placeholder="How can we help you?"
                  style={{
                    width: '100%',
                    padding: '16px 20px',
                    fontSize: '1rem',
                    background: 'rgba(255, 255, 255, 0.05)',
                    border: `2px solid ${errors.subject && touched.subject ? theme.colors.error : focusedInput === 'subject' ? theme.colors.primary : 'rgba(255, 255, 255, 0.1)'}`,
                    borderRadius: '14px',
                    outline: 'none',
                    transition: 'all 0.3s ease',
                    color: 'white',
                    boxShadow: focusedInput === 'subject' ? `0 0 0 4px ${theme.colors.primary}20` : 'none'
                  }}
                />
                {errors.subject && touched.subject && (
                  <div style={{
                    color: theme.colors.error,
                    fontSize: '0.85rem',
                    marginTop: '8px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px'
                  }}>
                    <AlertCircle size={14} />
                    {errors.subject}
                  </div>
                )}
              </div>

              {/* Message Input */}
              <div style={{ marginBottom: '32px' }}>
                <label style={{
                  display: 'block',
                  fontSize: '0.9rem',
                  fontWeight: '600',
                  color: '#e2e8f0',
                  marginBottom: '10px',
                  letterSpacing: '0.3px'
                }}>
                  Message *
                </label>
                <textarea
                  name="message"
                  value={formData.message}
                  onChange={handleChange}
                  onBlur={handleBlur}
                  onFocus={() => setFocusedInput('message')}
                  placeholder="Tell us more about your inquiry..."
                  rows={5}
                  style={{
                    width: '100%',
                    padding: '16px 20px',
                    fontSize: '1rem',
                    background: 'rgba(255, 255, 255, 0.05)',
                    border: `2px solid ${errors.message && touched.message ? theme.colors.error : focusedInput === 'message' ? theme.colors.primary : 'rgba(255, 255, 255, 0.1)'}`,
                    borderRadius: '14px',
                    outline: 'none',
                    transition: 'all 0.3s ease',
                    color: 'white',
                    resize: 'vertical',
                    minHeight: '140px',
                    boxShadow: focusedInput === 'message' ? `0 0 0 4px ${theme.colors.primary}20` : 'none'
                  }}
                />
                {errors.message && touched.message && (
                  <div style={{
                    color: theme.colors.error,
                    fontSize: '0.85rem',
                    marginTop: '8px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px'
                  }}>
                    <AlertCircle size={14} />
                    {errors.message}
                  </div>
                )}
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isSubmitting}
                style={{
                  width: '100%',
                  padding: '18px 32px',
                  fontSize: '1.05rem',
                  fontWeight: '700',
                  color: 'white',
                  background: isSubmitting ? '#64748b' : theme.gradients.purple,
                  border: 'none',
                  borderRadius: '14px',
                  cursor: isSubmitting ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '12px',
                  transition: 'all 0.3s ease',
                  boxShadow: isSubmitting ? 'none' : '0 10px 30px rgba(102, 126, 234, 0.4)',
                  opacity: isSubmitting ? 0.7 : 1,
                  transform: isSubmitting ? 'scale(0.98)' : 'scale(1)',
                  letterSpacing: '0.5px'
                }}
                onMouseEnter={(e) => {
                  if (!isSubmitting) {
                    e.currentTarget.style.transform = 'translateY(-2px) scale(1.02)';
                    e.currentTarget.style.boxShadow = '0 15px 40px rgba(102, 126, 234, 0.5)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isSubmitting) {
                    e.currentTarget.style.transform = 'scale(1)';
                    e.currentTarget.style.boxShadow = '0 10px 30px rgba(102, 126, 234, 0.4)';
                  }
                }}
              >
                {isSubmitting ? (
                  <>
                    Sending Message...
                    <div style={{
                      width: '20px',
                      height: '20px',
                      border: '2px solid white',
                      borderTopColor: 'transparent',
                      borderRadius: '50%',
                      animation: 'spin 1s linear infinite'
                    }} />
                  </>
                ) : (
                  <>
                    Send Message
                    <Send size={20} />
                  </>
                )}
              </button>
            </form>
          </div>

          {/* Right Column - Map & Social */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            {/* Map Card */}
            <div style={{
              background: 'rgba(255, 255, 255, 0.05)',
              backdropFilter: 'blur(20px)',
              borderRadius: '32px',
              padding: '40px',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              opacity: visible ? 1 : 0,
              transform: visible ? 'translateX(0)' : 'translateX(30px)',
              transition: 'all 0.8s cubic-bezier(0.34, 1.56, 0.64, 1) 0.4s'
            }}>
              <div style={{ marginBottom: '24px' }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '800',
                  color: 'white',
                  marginBottom: '8px'
                }}>
                  Visit Our Office
                </h3>
                <p style={{
                  fontSize: '0.95rem',
                  color: '#94a3b8',
                  lineHeight: 1.6
                }}>
                  Come meet us at our location in Patna, Bihar
                </p>
              </div>

              <div style={{
                width: '100%',
                height: '280px',
                borderRadius: '20px',
                background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '20px',
                border: '2px dashed rgba(255, 255, 255, 0.2)',
                position: 'relative',
                overflow: 'hidden'
              }}>
                <div style={{
                  position: 'absolute',
                  top: -50,
                  right: -50,
                  width: 150,
                  height: 150,
                  borderRadius: '50%',
                  background: 'rgba(102, 126, 234, 0.2)',
                  filter: 'blur(40px)'
                }} />
                
                <MapPin size={56} color="#667eea" style={{ animation: 'float 3s ease-in-out infinite', zIndex: 1 }} />
                
                <div style={{ textAlign: 'center', zIndex: 1 }}>
                  <div style={{
                    fontSize: '1.25rem',
                    fontWeight: '700',
                    color: 'white',
                    marginBottom: '8px'
                  }}>
                    Fraser Road Area
                  </div>
                  <div style={{
                    fontSize: '1rem',
                    color: '#cbd5e1',
                    marginBottom: '16px'
                  }}>
                    Patna, Bihar 800001, India
                  </div>
                  <button
                    style={{
                      padding: '12px 28px',
                      background: theme.gradients.purple,
                      color: 'white',
                      border: 'none',
                      borderRadius: '10px',
                      cursor: 'pointer',
                      fontWeight: '700',
                      fontSize: '0.95rem',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '8px',
                      transition: 'all 0.3s ease',
                      boxShadow: '0 8px 20px rgba(102, 126, 234, 0.4)'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.transform = 'translateY(-2px)';
                      e.currentTarget.style.boxShadow = '0 12px 30px rgba(102, 126, 234, 0.5)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.transform = 'translateY(0)';
                      e.currentTarget.style.boxShadow = '0 8px 20px rgba(102, 126, 234, 0.4)';
                    }}
                  >
                    <Map size={18} />
                    Open in Maps
                  </button>
                </div>
              </div>
            </div>

            {/* Social Media Card */}
            <div style={{
              background: 'rgba(255, 255, 255, 0.05)',
              backdropFilter: 'blur(20px)',
              borderRadius: '32px',
              padding: '40px',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              opacity: visible ? 1 : 0,
              transform: visible ? 'translateX(0)' : 'translateX(30px)',
              transition: 'all 0.8s cubic-bezier(0.34, 1.56, 0.64, 1) 0.5s'
            }}>
              <div style={{ marginBottom: '24px' }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '800',
                  color: 'white',
                  marginBottom: '8px'
                }}>
                  Follow Us
                </h3>
                <p style={{
                  fontSize: '0.95rem',
                  color: '#94a3b8',
                  lineHeight: 1.6
                }}>
                  Stay connected on social media
                </p>
              </div>

              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(2, 1fr)',
                gap: '16px'
              }}>
                {socialMedia.map((social, index) => (
                  <a
                    key={index}
                    href={social.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      padding: '20px',
                      background: hoveredSocial === index 
                        ? `linear-gradient(135deg, ${social.color}20 0%, ${social.color}10 100%)`
                        : 'rgba(255, 255, 255, 0.05)',
                      border: `2px solid ${hoveredSocial === index ? social.color : 'rgba(255, 255, 255, 0.1)'}`,
                      borderRadius: '16px',
                      textDecoration: 'none',
                      transition: 'all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)',
                      transform: hoveredSocial === index ? 'translateY(-8px) scale(1.05)' : 'translateY(0)',
                      boxShadow: hoveredSocial === index ? `0 12px 30px ${social.color}40` : 'none',
                      cursor: 'pointer'
                    }}
                    onMouseEnter={() => setHoveredSocial(index)}
                    onMouseLeave={() => setHoveredSocial(null)}
                  >
                    <div style={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      gap: '8px'
                    }}>
                      <div style={{ color: hoveredSocial === index ? social.color : '#94a3b8' }}>
                        {React.cloneElement(social.icon, { size: 28 })}
                      </div>
                      <span style={{
                        fontSize: '0.85rem',
                        fontWeight: '700',
                        color: hoveredSocial === index ? social.color : '#cbd5e1'
                      }}>
                        {social.name}
                      </span>
                    </div>
                  </a>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Stats Section */}
        <div style={{
          marginTop: '80px',
          padding: '60px 40px',
          background: 'rgba(255, 255, 255, 0.03)',
          backdropFilter: 'blur(20px)',
          borderRadius: '32px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          position: 'relative',
          overflow: 'hidden'
        }}>
          <div style={{
            position: 'absolute',
            top: -100,
            left: -100,
            width: 300,
            height: 300,
            borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(102, 126, 234, 0.3) 0%, transparent 70%)',
            filter: 'blur(60px)',
            pointerEvents: 'none'
          }} />

          <div style={{
            textAlign: 'center',
            marginBottom: '50px',
            position: 'relative',
            zIndex: 1
          }}>
            <h3 style={{
              fontSize: 'clamp(1.8rem, 3vw, 2.5rem)',
              fontWeight: '800',
              color: 'white',
              marginBottom: '12px'
            }}>
              Why Choose MeenaSetu?
            </h3>
            <p style={{
              fontSize: '1.1rem',
              color: '#94a3b8',
              maxWidth: '600px',
              margin: '0 auto'
            }}>
              Trusted by hundreds of clients for exceptional AI-powered fish health solutions
            </p>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
            gap: '32px',
            position: 'relative',
            zIndex: 1
          }}>
            {stats.map((stat, idx) => (
              <div
                key={idx}
                style={{
                  textAlign: 'center',
                  padding: '40px 24px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  backdropFilter: 'blur(10px)',
                  borderRadius: '24px',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  transition: 'all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)',
                  cursor: 'pointer'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-12px) scale(1.03)';
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                  e.currentTarget.style.boxShadow = '0 20px 50px rgba(0, 0, 0, 0.3)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0) scale(1)';
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                  e.currentTarget.style.boxShadow = 'none';
                }}
              >
                <div style={{
                  width: '80px',
                  height: '80px',
                  margin: '0 auto 24px',
                  borderRadius: '20px',
                  background: stat.gradient,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                  fontSize: '2rem',
                  boxShadow: '0 10px 30px rgba(0, 0, 0, 0.2)'
                }}>
                  {React.cloneElement(stat.icon, { size: 36 })}
                </div>
                <div style={{
                  fontSize: '3rem',
                  fontWeight: '900',
                  background: stat.gradient,
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  marginBottom: '12px',
                  lineHeight: 1
                }}>
                  {stat.value}
                </div>
                <div style={{
                  fontSize: '1rem',
                  color: '#cbd5e1',
                  fontWeight: '600',
                  letterSpacing: '0.5px'
                }}>
                  {stat.label}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Call to Action */}
        <div style={{
          marginTop: '60px',
          padding: '50px 40px',
          background: theme.gradients.purple,
          borderRadius: '32px',
          textAlign: 'center',
          position: 'relative',
          overflow: 'hidden'
        }}>
          <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'url("data:image/svg+xml,%3Csvg width="60" height="60" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="none" fill-rule="evenodd"%3E%3Cg fill="%23ffffff" fill-opacity="0.05"%3E%3Cpath d="M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z"/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")',
            opacity: 0.3,
            pointerEvents: 'none'
          }} />

          <div style={{ position: 'relative', zIndex: 1 }}>
            <h3 style={{
              fontSize: 'clamp(1.5rem, 3vw, 2rem)',
              fontWeight: '800',
              color: 'white',
              marginBottom: '16px'
            }}>
              Ready to Get Started?
            </h3>
            <p style={{
              fontSize: '1.1rem',
              color: 'rgba(255, 255, 255, 0.9)',
              marginBottom: '32px',
              maxWidth: '600px',
              margin: '0 auto 32px'
            }}>
              Join hundreds of satisfied clients who trust MeenaSetu for their fish health needs
            </p>
            <button
              style={{
                padding: '18px 40px',
                background: 'white',
                color: theme.colors.primary,
                border: 'none',
                borderRadius: '14px',
                fontSize: '1.1rem',
                fontWeight: '700',
                cursor: 'pointer',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '12px',
                transition: 'all 0.3s ease',
                boxShadow: '0 10px 30px rgba(0, 0, 0, 0.2)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-4px) scale(1.05)';
                e.currentTarget.style.boxShadow = '0 15px 40px rgba(0, 0, 0, 0.3)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0) scale(1)';
                e.currentTarget.style.boxShadow = '0 10px 30px rgba(0, 0, 0, 0.2)';
              }}
            >
              Start Your Journey
              <ArrowRight size={22} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Contact;