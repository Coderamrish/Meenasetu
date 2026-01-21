import React, { useState, useEffect, useCallback } from 'react';
import { 
  Mail, Phone, MapPin, Send, Facebook, Twitter, Linkedin, Instagram, 
  Clock, CheckCircle, AlertCircle, Map, Building, Users, Globe 
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
  const [animateSuccess, setAnimateSuccess] = useState(false);

  useEffect(() => {
    setVisible(true);
  }, []);

  // Theme configuration
  const theme = {
    primary: '#2563eb',
    primaryDark: '#1d4ed8',
    primaryLight: '#60a5fa',
    secondary: '#7c3aed',
    secondaryDark: '#5b21b6',
    background: '#f8fafc',
    paper: '#ffffff',
    text: '#1e293b',
    textSecondary: '#64748b',
    border: '#e2e8f0',
    success: '#059669',
    error: '#dc2626',
    warning: '#d97706',
    info: '#0ea5e9'
  };

  // Memoized validation function
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
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        setIsSubmitting(false);
        setSubmitSuccess(true);
        setAnimateSuccess(true);
        setFormData({ name: '', email: '', subject: '', message: '' });
        setTouched({});
        
        // Reset success animation after 5 seconds
        setTimeout(() => {
          setSubmitSuccess(false);
          setAnimateSuccess(false);
        }, 5000);
      } catch (error) {
        setIsSubmitting(false);
        // Handle error state here
      }
    }
  }, [formData, validateField]);

  // Contact information
  const contactInfo = useCallback(() => [
    {
      icon: <Phone size={32} />,
      title: 'Phone',
      primary: '+91 1234 567 890',
      secondary: 'Mon-Fri 9am-6pm',
      link: 'tel:+911234567890',
      color: theme.primary,
      delay: '0s'
    },
    {
      icon: <Mail size={32} />,
      title: 'Email',
      primary: 'contact@meenasetu.com',
      secondary: 'Online support',
      link: 'mailto:contact@meenasetu.com',
      color: theme.secondary,
      delay: '0.1s'
    },
    {
      icon: <MapPin size={32} />,
      title: 'Address',
      primary: 'Fraser Road Area',
      secondary: 'Patna, Bihar 800001',
      link: '#map',
      color: theme.success,
      delay: '0.2s'
    },
    {
      icon: <Clock size={32} />,
      title: 'Business Hours',
      primary: 'Mon-Fri: 9:00 AM - 6:00 PM',
      secondary: 'Sat: 10:00 AM - 4:00 PM',
      link: '#',
      color: theme.warning,
      delay: '0.3s'
    }
  ], [theme]);

  // Social media links
  const socialMedia = useCallback(() => [
    { icon: <Facebook size={24} />, name: 'Facebook', link: '#', color: '#1877f2' },
    { icon: <Twitter size={24} />, name: 'Twitter', link: '#', color: '#1da1f2' },
    { icon: <Linkedin size={24} />, name: 'LinkedIn', link: '#', color: '#0a66c2' },
    { icon: <Instagram size={24} />, name: 'Instagram', link: '#', color: '#e4405f' }
  ], []);

  // Styles
  const styles = {
    container: {
      minHeight: '100vh',
      background: `linear-gradient(135deg, ${theme.background} 0%, ${theme.primary}05 100%)`,
      fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
      padding: 0,
      margin: 0,
      position: 'relative',
      overflow: 'hidden'
    },
    heroSection: {
      background: `linear-gradient(135deg, ${theme.primary} 0%, ${theme.secondary} 100%)`,
      color: 'white',
      padding: '100px 20px',
      textAlign: 'center',
      position: 'relative',
      overflow: 'hidden',
      opacity: visible ? 1 : 0,
      transform: visible ? 'translateY(0)' : 'translateY(-20px)',
      transition: 'all 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)',
      clipPath: visible ? 'polygon(0 0, 100% 0, 100% 100%, 0 95%)' : 'polygon(0 0, 100% 0, 100% 0, 0 0)'
    },
    heroPattern: {
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundImage: `
        radial-gradient(circle at 10% 20%, rgba(255,255,255,0.1) 0%, transparent 20%),
        radial-gradient(circle at 90% 80%, rgba(255,255,255,0.1) 0%, transparent 20%),
        linear-gradient(45deg, transparent 49%, rgba(255,255,255,0.05) 50%, transparent 51%)
      `,
      backgroundSize: '50px 50px',
      pointerEvents: 'none'
    },
    heroContent: {
      position: 'relative',
      zIndex: 1,
      maxWidth: '800px',
      margin: '0 auto'
    },
    heroTitle: {
      fontSize: 'clamp(2rem, 5vw, 3.5rem)',
      fontWeight: 800,
      marginBottom: '20px',
      letterSpacing: '-0.5px',
      lineHeight: 1.2
    },
    heroSubtitle: {
      fontSize: 'clamp(1rem, 2vw, 1.25rem)',
      opacity: 0.95,
      fontWeight: 300,
      lineHeight: 1.6,
      maxWidth: '600px',
      margin: '0 auto'
    },
    mainContent: {
      maxWidth: '1280px',
      margin: '-80px auto 0',
      padding: '0 20px 100px',
      position: 'relative',
      zIndex: 2
    },
    infoCardsGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
      gap: '30px',
      marginBottom: '60px'
    },
    infoCard: {
      background: 'white',
      borderRadius: '20px',
      padding: '40px 30px',
      boxShadow: '0 10px 40px rgba(0,0,0,0.08)',
      transition: 'all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)',
      cursor: 'pointer',
      position: 'relative',
      overflow: 'hidden',
      opacity: visible ? 1 : 0,
      transform: visible ? 'translateY(0)' : 'translateY(30px)',
      border: '1px solid transparent'
    },
    cardIconWrapper: {
      width: '72px',
      height: '72px',
      borderRadius: '20px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      marginBottom: '24px',
      transition: 'all 0.3s ease'
    },
    cardTitle: {
      fontSize: '0.875rem',
      fontWeight: 600,
      color: theme.textSecondary,
      marginBottom: '12px',
      textTransform: 'uppercase',
      letterSpacing: '1px'
    },
    cardPrimary: {
      fontSize: '1.25rem',
      fontWeight: 700,
      color: theme.text,
      marginBottom: '8px',
      lineHeight: 1.3
    },
    cardSecondary: {
      fontSize: '0.9375rem',
      color: theme.textSecondary,
      lineHeight: 1.5
    },
    contentGrid: {
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      gap: '48px',
      alignItems: 'start'
    },
    formCard: {
      background: 'white',
      borderRadius: '24px',
      padding: '48px',
      boxShadow: '0 20px 60px rgba(0,0,0,0.1)',
      opacity: visible ? 1 : 0,
      transform: visible ? 'translateX(0)' : 'translateX(-30px)',
      transition: 'all 0.8s cubic-bezier(0.34, 1.56, 0.64, 1) 0.2s',
      border: '1px solid rgba(0,0,0,0.05)'
    },
    sectionTitle: {
      fontSize: 'clamp(1.5rem, 3vw, 2.25rem)',
      fontWeight: 800,
      color: theme.text,
      marginBottom: '12px',
      lineHeight: 1.2
    },
    sectionSubtitle: {
      fontSize: '1.0625rem',
      color: theme.textSecondary,
      marginBottom: '40px',
      lineHeight: 1.6
    },
    inputWrapper: {
      marginBottom: '28px',
      position: 'relative'
    },
    label: {
      display: 'block',
      fontSize: '0.9375rem',
      fontWeight: 600,
      color: theme.text,
      marginBottom: '10px'
    },
    input: {
      width: '100%',
      padding: '16px 20px',
      fontSize: '1rem',
      border: `1px solid ${theme.border}`,
      borderRadius: '12px',
      outline: 'none',
      transition: 'all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)',
      fontFamily: 'inherit',
      boxSizing: 'border-box',
      backgroundColor: 'white',
      '&::placeholder': {
        color: theme.textSecondary
      }
    },
    textarea: {
      width: '100%',
      padding: '16px 20px',
      fontSize: '1rem',
      border: `1px solid ${theme.border}`,
      borderRadius: '12px',
      outline: 'none',
      transition: 'all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)',
      fontFamily: 'inherit',
      minHeight: '160px',
      resize: 'vertical',
      boxSizing: 'border-box',
      backgroundColor: 'white',
      '&::placeholder': {
        color: theme.textSecondary
      }
    },
    errorText: {
      color: theme.error,
      fontSize: '0.8125rem',
      marginTop: '8px',
      display: 'flex',
      alignItems: 'center',
      gap: '6px',
      fontWeight: 500
    },
    button: {
      width: '100%',
      padding: '18px 32px',
      fontSize: '1rem',
      fontWeight: 700,
      color: 'white',
      background: `linear-gradient(135deg, ${theme.primary} 0%, ${theme.secondary} 100%)`,
      border: 'none',
      borderRadius: '12px',
      cursor: 'pointer',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '12px',
      transition: 'all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)',
      boxShadow: `0 10px 30px ${theme.primary}40`,
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
      position: 'relative',
      overflow: 'hidden',
      '&::before': {
        content: '""',
        position: 'absolute',
        top: 0,
        left: '-100%',
        width: '100%',
        height: '100%',
        background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent)',
        transition: 'left 0.7s ease'
      }
    },
    mapCard: {
      background: 'white',
      borderRadius: '24px',
      padding: '40px',
      boxShadow: '0 20px 60px rgba(0,0,0,0.1)',
      marginBottom: '40px',
      opacity: visible ? 1 : 0,
      transform: visible ? 'translateX(0)' : 'translateX(30px)',
      transition: 'all 0.8s cubic-bezier(0.34, 1.56, 0.64, 1) 0.3s',
      border: '1px solid rgba(0,0,0,0.05)'
    },
    mapContainer: {
      width: '100%',
      height: '320px',
      borderRadius: '16px',
      overflow: 'hidden',
      marginTop: '24px',
      background: `linear-gradient(135deg, ${theme.background} 0%, ${theme.primary}10 100%)`,
      position: 'relative'
    },
    mapPlaceholder: {
      width: '100%',
      height: '100%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      flexDirection: 'column',
      gap: '16px',
      color: theme.textSecondary,
      padding: '20px'
    },
    socialCard: {
      background: 'white',
      borderRadius: '24px',
      padding: '40px',
      boxShadow: '0 20px 60px rgba(0,0,0,0.1)',
      opacity: visible ? 1 : 0,
      transform: visible ? 'translateX(0)' : 'translateX(30px)',
      transition: 'all 0.8s cubic-bezier(0.34, 1.56, 0.64, 1) 0.4s',
      border: '1px solid rgba(0,0,0,0.05)'
    },
    socialGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(4, 1fr)',
      gap: '20px',
      marginTop: '28px'
    },
    socialButton: {
      aspectRatio: '1',
      borderRadius: '16px',
      border: `2px solid ${theme.border}`,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      cursor: 'pointer',
      transition: 'all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)',
      backgroundColor: 'white',
      textDecoration: 'none'
    },
    successAlert: {
      position: 'fixed',
      top: '30px',
      right: '30px',
      background: theme.success,
      color: 'white',
      padding: '20px 28px',
      borderRadius: '12px',
      boxShadow: '0 20px 40px rgba(0,0,0,0.2)',
      display: 'flex',
      alignItems: 'center',
      gap: '14px',
      zIndex: 1000,
      animation: animateSuccess ? 'slideIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)' : 'none',
      borderLeft: `4px solid ${theme.success}`
    },
    statsContainer: {
      display: 'grid',
      gridTemplateColumns: 'repeat(3, 1fr)',
      gap: '24px',
      marginTop: '60px',
      paddingTop: '60px',
      borderTop: `1px solid ${theme.border}`
    },
    statCard: {
      textAlign: 'center',
      padding: '30px 20px',
      background: 'white',
      borderRadius: '16px',
      boxShadow: '0 10px 30px rgba(0,0,0,0.05)',
      transition: 'all 0.3s ease'
    }
  };

  const getInputStyle = (fieldName) => ({
    ...styles.input,
    borderColor: errors[fieldName] && touched[fieldName] ? theme.error : 
                 focusedInput === fieldName ? theme.primary : theme.border,
    borderWidth: focusedInput === fieldName ? '2px' : '1px',
    boxShadow: focusedInput === fieldName ? `0 0 0 4px ${theme.primary}15` : 'none',
    backgroundColor: focusedInput === fieldName ? `${theme.primary}05` : 'white'
  });

  const getTextareaStyle = () => ({
    ...styles.textarea,
    borderColor: errors.message && touched.message ? theme.error : 
                 focusedInput === 'message' ? theme.primary : theme.border,
    borderWidth: focusedInput === 'message' ? '2px' : '1px',
    boxShadow: focusedInput === 'message' ? `0 0 0 4px ${theme.primary}15` : 'none',
    backgroundColor: focusedInput === 'message' ? `${theme.primary}05` : 'white'
  });

  return (
    <div style={styles.container}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        @keyframes slideIn {
          0% { transform: translateX(400px) scale(0.9); opacity: 0; }
          100% { transform: translateX(0) scale(1); opacity: 1; }
        }
        
        @keyframes float {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-10px); }
        }
        
        @keyframes shimmer {
          0% { left: -100%; }
          100% { left: 100%; }
        }
        
        * {
          box-sizing: border-box;
        }
        
        button:hover::before {
          left: 100%;
        }
        
        @media (max-width: 1024px) {
          .content-grid { grid-template-columns: 1fr !important; }
          .hero-section { padding: 80px 20px !important; }
          .info-cards-grid { grid-template-columns: repeat(2, 1fr) !important; }
        }
        
        @media (max-width: 768px) {
          .hero-title { font-size: 2rem !important; }
          .info-cards-grid { grid-template-columns: 1fr !important; }
          .social-grid { grid-template-columns: repeat(2, 1fr) !important; }
          .main-content { margin-top: -60px !important; padding-bottom: 60px !important; }
          .form-card { padding: 32px !important; }
          .map-card, .social-card { padding: 32px !important; }
        }
        
        @media (max-width: 480px) {
          .hero-section { padding: 60px 16px !important; }
          .hero-title { font-size: 1.75rem !important; }
          .section-title { font-size: 1.5rem !important; }
          .success-alert {
            top: 16px;
            right: 16px;
            left: 16px;
            padding: 16px;
          }
        }
      `}</style>

      {submitSuccess && (
        <div style={styles.successAlert} className="success-alert">
          <CheckCircle size={28} />
          <div>
            <div style={{ fontWeight: 700, marginBottom: '4px' }}>Success!</div>
            <div style={{ fontSize: '0.9375rem', opacity: 0.9 }}>
              Message sent successfully! We'll get back to you within 24 hours.
            </div>
          </div>
        </div>
      )}

      <div style={styles.heroSection} className="hero-section">
        <div style={styles.heroPattern} />
        <div style={styles.heroContent}>
          <h1 style={styles.heroTitle} className="hero-title">
            Connect With <span style={{ color: '#fbbf24' }}>MeenaSetu</span>
          </h1>
          <p style={styles.heroSubtitle}>
            Have questions or need assistance? Our dedicated team is ready to help you 
            with any inquiries. We're committed to providing exceptional support and 
            building lasting relationships.
          </p>
        </div>
      </div>

      <div style={styles.mainContent}>
        <div style={styles.infoCardsGrid} className="info-cards-grid">
          {contactInfo().map((info, index) => (
            <a
              key={index}
              href={info.link}
              style={{
                ...styles.infoCard,
                textDecoration: 'none',
                transform: hoveredCard === index ? 'translateY(-12px) scale(1.02)' : 
                          visible ? 'translateY(0)' : 'translateY(30px)',
                boxShadow: hoveredCard === index ? `0 20px 60px ${info.color}20` : '0 10px 40px rgba(0,0,0,0.08)',
                borderColor: hoveredCard === index ? `${info.color}30` : 'transparent',
                transitionDelay: info.delay
              }}
              onMouseEnter={() => setHoveredCard(index)}
              onMouseLeave={() => setHoveredCard(null)}
              onFocus={() => setHoveredCard(index)}
              onBlur={() => setHoveredCard(null)}
            >
              <div style={{
                ...styles.cardIconWrapper,
                background: `linear-gradient(135deg, ${info.color}15 0%, ${info.color}05 100%)`,
                color: info.color,
                transform: hoveredCard === index ? 'scale(1.1) rotate(5deg)' : 'scale(1)',
                boxShadow: `0 10px 30px ${info.color}15`
              }}>
                {info.icon}
              </div>
              <div style={styles.cardTitle}>{info.title}</div>
              <div style={styles.cardPrimary}>{info.primary}</div>
              <div style={styles.cardSecondary}>{info.secondary}</div>
            </a>
          ))}
        </div>

        <div style={styles.contentGrid} className="content-grid">
          <div style={styles.formCard} className="form-card">
            <h2 style={styles.sectionTitle}>Send Us a Message</h2>
            <p style={styles.sectionSubtitle}>
              Fill out the form below and we'll get back to you as soon as possible. 
              All fields marked with * are required.
            </p>

            <div style={styles.inputWrapper}>
              <label style={styles.label}>Full Name *</label>
              <input
                type="text"
                name="name"
                value={formData.name}
                onChange={handleChange}
                onBlur={handleBlur}
                onFocus={() => setFocusedInput('name')}
                style={getInputStyle('name')}
                placeholder="John Doe"
                aria-label="Full Name"
                aria-required="true"
                aria-invalid={!!errors.name && touched.name}
              />
              {errors.name && touched.name && (
                <div style={styles.errorText}>
                  <AlertCircle size={16} />
                  {errors.name}
                </div>
              )}
            </div>

            <div style={styles.inputWrapper}>
              <label style={styles.label}>Email Address *</label>
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                onBlur={handleBlur}
                onFocus={() => setFocusedInput('email')}
                style={getInputStyle('email')}
                placeholder="john@example.com"
                aria-label="Email Address"
                aria-required="true"
                aria-invalid={!!errors.email && touched.email}
              />
              {errors.email && touched.email && (
                <div style={styles.errorText}>
                  <AlertCircle size={16} />
                  {errors.email}
                </div>
              )}
            </div>

            <div style={styles.inputWrapper}>
              <label style={styles.label}>Subject *</label>
              <input
                type="text"
                name="subject"
                value={formData.subject}
                onChange={handleChange}
                onBlur={handleBlur}
                onFocus={() => setFocusedInput('subject')}
                style={getInputStyle('subject')}
                placeholder="How can we help you?"
                aria-label="Subject"
                aria-required="true"
                aria-invalid={!!errors.subject && touched.subject}
              />
              {errors.subject && touched.subject && (
                <div style={styles.errorText}>
                  <AlertCircle size={16} />
                  {errors.subject}
                </div>
              )}
            </div>

            <div style={styles.inputWrapper}>
              <label style={styles.label}>Message *</label>
              <textarea
                name="message"
                value={formData.message}
                onChange={handleChange}
                onBlur={handleBlur}
                onFocus={() => setFocusedInput('message')}
                style={getTextareaStyle()}
                placeholder="Tell us more about your inquiry, project, or question..."
                aria-label="Message"
                aria-required="true"
                aria-invalid={!!errors.message && touched.message}
              />
              {errors.message && touched.message && (
                <div style={styles.errorText}>
                  <AlertCircle size={16} />
                  {errors.message}
                </div>
              )}
            </div>

            <button
              onClick={handleSubmit}
              disabled={isSubmitting}
              style={{
                ...styles.button,
                opacity: isSubmitting ? 0.8 : 1,
                cursor: isSubmitting ? 'not-allowed' : 'pointer',
                transform: isSubmitting ? 'scale(0.98)' : 'scale(1)',
                background: isSubmitting ? theme.textSecondary : 
                           `linear-gradient(135deg, ${theme.primary} 0%, ${theme.secondary} 100%)`
              }}
              aria-label={isSubmitting ? 'Sending message' : 'Send message'}
            >
              {isSubmitting ? (
                <>
                  Sending...
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
                  <Send size={22} />
                </>
              )}
            </button>
          </div>

          <div>
            <div style={styles.mapCard} className="map-card">
              <h3 style={styles.sectionTitle}>Our Location</h3>
              <p style={styles.sectionSubtitle}>
                Visit us at our office in Patna, Bihar. We're located in the heart of the city.
              </p>
              
              <div style={styles.mapContainer}>
                <div style={styles.mapPlaceholder}>
                  <MapPin size={64} color={theme.primary} style={{ animation: 'float 3s ease-in-out infinite' }} />
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ 
                      fontSize: '1.25rem', 
                      fontWeight: 700, 
                      marginBottom: '8px',
                      color: theme.text 
                    }}>
                      Fraser Road Area
                    </div>
                    <div style={{ fontSize: '1rem', color: theme.textSecondary }}>
                      Patna, Bihar 800001, India
                    </div>
                    <button
                      style={{
                        marginTop: '20px',
                        padding: '10px 24px',
                        background: theme.primary,
                        color: 'white',
                        border: 'none',
                        borderRadius: '8px',
                        cursor: 'pointer',
                        fontWeight: 600,
                        fontSize: '0.9375rem',
                        transition: 'all 0.3s ease'
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-2px)'}
                      onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}
                    >
                      <Map size={16} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
                      Open in Maps
                    </button>
                  </div>
                </div>
              </div>
            </div>

            <div style={styles.socialCard} className="social-card">
              <h3 style={styles.sectionTitle}>Connect With Us</h3>
              <p style={styles.sectionSubtitle}>
                Follow us on social media to stay updated with our latest news, 
                announcements, and community activities.
              </p>
              
              <div style={styles.socialGrid} className="social-grid">
                {socialMedia().map((social, index) => (
                  <a
                    key={index}
                    href={social.link}
                    style={{
                      ...styles.socialButton,
                      borderColor: hoveredSocial === index ? social.color : theme.border,
                      background: hoveredSocial === index ? 
                                `linear-gradient(135deg, ${social.color}15 0%, ${social.color}05 100%)` : 
                                'white',
                      color: hoveredSocial === index ? social.color : theme.textSecondary,
                      transform: hoveredSocial === index ? 'translateY(-6px) scale(1.05)' : 'translateY(0)',
                      boxShadow: hoveredSocial === index ? `0 12px 30px ${social.color}30` : 'none'
                    }}
                    onMouseEnter={() => setHoveredSocial(index)}
                    onMouseLeave={() => setHoveredSocial(null)}
                    onFocus={() => setHoveredSocial(index)}
                    onBlur={() => setHoveredSocial(null)}
                    aria-label={`Visit our ${social.name} page`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    {social.icon}
                  </a>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div style={styles.statsContainer}>
          <div style={styles.statCard}>
            <Building size={48} color={theme.primary} style={{ marginBottom: '16px' }} />
            <div style={{ fontSize: '2rem', fontWeight: 800, color: theme.primary, marginBottom: '8px' }}>
              5+
            </div>
            <div style={{ color: theme.textSecondary, fontSize: '0.9375rem' }}>
              Years of Experience
            </div>
          </div>
          
          <div style={styles.statCard}>
            <Users size={48} color={theme.secondary} style={{ marginBottom: '16px' }} />
            <div style={{ fontSize: '2rem', fontWeight: 800, color: theme.secondary, marginBottom: '8px' }}>
              500+
            </div>
            <div style={{ color: theme.textSecondary, fontSize: '0.9375rem' }}>
              Happy Clients
            </div>
          </div>
          
          <div style={styles.statCard}>
            <Globe size={48} color={theme.success} style={{ marginBottom: '16px' }} />
            <div style={{ fontSize: '2rem', fontWeight: 800, color: theme.success, marginBottom: '8px' }}>
              24/7
            </div>
            <div style={{ color: theme.textSecondary, fontSize: '0.9375rem' }}>
              Support Available
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Contact;