import React, { useState, useEffect, useMemo } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline, alpha, GlobalStyles, Box, LinearProgress } from '@mui/material';
import { Toaster } from 'react-hot-toast';

// Layout Components
import MainLayout from './layouts/MainLayout';

// Pages
import Home from './pages/Home/Home';
import Analytics from './pages/Analytics/Analytics';
import DataExplorer from './pages/DataExplorer/DataExplorer';
import ChatbotPage from './pages/ChatbotPage/ChatbotPage';
import Profile from './pages/Profile/Profile';
import About from './pages/About/About';
import Contact from './pages/Contact/Contact';

// Advanced Loading Component with Logo Animation
const PageLoader = () => (
  <Box
    sx={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      flexDirection: 'column',
      background: 'linear-gradient(135deg, #0a1929 0%, #132f4c 50%, #0a1929 100%)',
      zIndex: 9999,
      overflow: 'hidden',
      '&::before': {
        content: '""',
        position: 'absolute',
        top: '-50%',
        left: '-50%',
        width: '200%',
        height: '200%',
        background: 'radial-gradient(circle, rgba(2,119,189,0.1) 0%, transparent 70%)',
        animation: 'rotate 20s linear infinite',
      },
      '@keyframes rotate': {
        '0%': { transform: 'rotate(0deg)' },
        '100%': { transform: 'rotate(360deg)' },
      },
    }}
  >
    {/* Animated Logo */}
    <Box
      sx={{
        position: 'relative',
        mb: 4,
        animation: 'pulse 2s ease-in-out infinite',
        '@keyframes pulse': {
          '0%, 100%': { transform: 'scale(1)', opacity: 0.9 },
          '50%': { transform: 'scale(1.05)', opacity: 1 },
        },
      }}
    >
      <Box
        sx={{
          fontSize: '5rem',
          background: 'linear-gradient(135deg, #0277bd 0%, #00897b 100%)',
          backgroundClip: 'text',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          fontWeight: 900,
          letterSpacing: '-0.05em',
          textShadow: '0 0 60px rgba(2,119,189,0.5)',
        }}
      >
        🌊
      </Box>
    </Box>

    {/* Spinner with Glow Effect */}
    <Box
      sx={{
        position: 'relative',
        width: '80px',
        height: '80px',
        mb: 3,
      }}
    >
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          border: '6px solid rgba(2,119,189,0.2)',
          borderTop: '6px solid #0277bd',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
          boxShadow: '0 0 30px rgba(2,119,189,0.6), inset 0 0 20px rgba(2,119,189,0.3)',
          '@keyframes spin': {
            '0%': { transform: 'rotate(0deg)' },
            '100%': { transform: 'rotate(360deg)' },
          },
        }}
      />
      <Box
        sx={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '50px',
          height: '50px',
          border: '4px solid rgba(0,137,123,0.3)',
          borderBottom: '4px solid #00897b',
          borderRadius: '50%',
          animation: 'spin-reverse 1.5s linear infinite',
          '@keyframes spin-reverse': {
            '0%': { transform: 'translate(-50%, -50%) rotate(0deg)' },
            '100%': { transform: 'translate(-50%, -50%) rotate(-360deg)' },
          },
        }}
      />
    </Box>

    {/* Loading Text with Gradient */}
    <Box
      sx={{
        fontSize: '1.5rem',
        fontWeight: 700,
        background: 'linear-gradient(135deg, #0277bd 0%, #00897b 100%)',
        backgroundClip: 'text',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
        mb: 2,
        animation: 'fadeInOut 2s ease-in-out infinite',
        '@keyframes fadeInOut': {
          '0%, 100%': { opacity: 0.6 },
          '50%': { opacity: 1 },
        },
      }}
    >
      MeenaSetu AI
    </Box>

    {/* Progress Bar */}
    <Box sx={{ width: '300px', px: 2 }}>
      <LinearProgress
        sx={{
          height: 6,
          borderRadius: 3,
          bgcolor: 'rgba(2,119,189,0.2)',
          '& .MuiLinearProgress-bar': {
            background: 'linear-gradient(90deg, #0277bd 0%, #00897b 100%)',
            borderRadius: 3,
            boxShadow: '0 0 20px rgba(2,119,189,0.6)',
          },
        }}
      />
    </Box>

    {/* Subtitle */}
    <Box
      sx={{
        mt: 2,
        color: 'rgba(255,255,255,0.6)',
        fontSize: '0.9rem',
        fontWeight: 500,
        letterSpacing: '0.1em',
      }}
    >
      Aquatic Intelligence Platform
    </Box>

    {/* Floating Particles */}
    {[...Array(20)].map((_, i) => (
      <Box
        key={i}
        sx={{
          position: 'absolute',
          width: `${Math.random() * 6 + 2}px`,
          height: `${Math.random() * 6 + 2}px`,
          background: 'rgba(2,119,189,0.6)',
          borderRadius: '50%',
          top: `${Math.random() * 100}%`,
          left: `${Math.random() * 100}%`,
          animation: `float ${Math.random() * 10 + 10}s ease-in-out infinite`,
          animationDelay: `${Math.random() * 5}s`,
          boxShadow: '0 0 10px rgba(2,119,189,0.8)',
          '@keyframes float': {
            '0%, 100%': {
              transform: 'translateY(0) translateX(0)',
              opacity: 0,
            },
            '25%': {
              opacity: 0.8,
            },
            '50%': {
              transform: `translateY(-${Math.random() * 100 + 50}px) translateX(${Math.random() * 100 - 50}px)`,
              opacity: 1,
            },
            '75%': {
              opacity: 0.6,
            },
          },
        }}
      />
    ))}
  </Box>
);

function App() {
  const [loading, setLoading] = useState(true);
  const [darkMode, setDarkMode] = useState(() => {
    const savedMode = localStorage.getItem('themeMode');
    return savedMode === 'dark';
  });

  useEffect(() => {
    // Enhanced loading with minimum display time
    const timer = setTimeout(() => setLoading(false), 2000);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    localStorage.setItem('themeMode', darkMode ? 'dark' : 'light');
    
    // Add smooth transition class to body
    document.body.style.transition = 'background-color 0.3s ease, color 0.3s ease';
  }, [darkMode]);

  // Enhanced Dynamic Theme with Advanced Customization
  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          mode: darkMode ? 'dark' : 'light',
          primary: {
            main: darkMode ? '#3b82f6' : '#0277bd',
            light: darkMode ? '#60a5fa' : '#58a5f0',
            dark: darkMode ? '#2563eb' : '#004c8c',
            contrastText: '#ffffff',
          },
          secondary: {
            main: darkMode ? '#10b981' : '#00897b',
            light: darkMode ? '#34d399' : '#4ebaaa',
            dark: darkMode ? '#059669' : '#005b4f',
            contrastText: '#ffffff',
          },
          background: {
            default: darkMode ? '#0a1929' : '#f8f9fa',
            paper: darkMode ? '#132f4c' : '#ffffff',
          },
          success: {
            main: '#10b981',
            light: '#34d399',
            dark: '#059669',
          },
          error: {
            main: '#ef4444',
            light: '#f87171',
            dark: '#dc2626',
          },
          warning: {
            main: '#f59e0b',
            light: '#fbbf24',
            dark: '#d97706',
          },
          info: {
            main: '#3b82f6',
            light: '#60a5fa',
            dark: '#2563eb',
          },
          gradient: {
            primary: darkMode 
              ? 'linear-gradient(135deg, #1e3a5f 0%, #1a5f5f 100%)'
              : 'linear-gradient(135deg, #0277bd 0%, #00897b 100%)',
            secondary: darkMode
              ? 'linear-gradient(135deg, #5b21b6 0%, #7c3aed 100%)'
              : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            success: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
            warning: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
            info: 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)',
            dark: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
          },
        },
        typography: {
          fontFamily: '"Inter", "Roboto", "Segoe UI", "Helvetica Neue", sans-serif',
          h1: {
            fontSize: '2.75rem',
            fontWeight: 800,
            letterSpacing: '-0.02em',
            lineHeight: 1.2,
          },
          h2: {
            fontSize: '2.25rem',
            fontWeight: 700,
            letterSpacing: '-0.01em',
            lineHeight: 1.3,
          },
          h3: {
            fontSize: '1.875rem',
            fontWeight: 700,
            letterSpacing: '-0.01em',
            lineHeight: 1.3,
          },
          h4: {
            fontSize: '1.5rem',
            fontWeight: 600,
            letterSpacing: '-0.005em',
            lineHeight: 1.4,
          },
          h5: {
            fontSize: '1.25rem',
            fontWeight: 600,
            letterSpacing: '0em',
            lineHeight: 1.4,
          },
          h6: {
            fontSize: '1rem',
            fontWeight: 600,
            letterSpacing: '0em',
            lineHeight: 1.5,
          },
          body1: {
            fontSize: '1rem',
            lineHeight: 1.6,
            letterSpacing: '0.00938em',
          },
          body2: {
            fontSize: '0.875rem',
            lineHeight: 1.5,
            letterSpacing: '0.01071em',
          },
          button: {
            fontWeight: 600,
            letterSpacing: '0.02857em',
            textTransform: 'none',
          },
        },
        shape: {
          borderRadius: 12,
        },
        shadows: darkMode
          ? [
              'none',
              '0 1px 3px rgba(0,0,0,0.3)',
              '0 4px 6px rgba(0,0,0,0.4)',
              '0 5px 15px rgba(0,0,0,0.5)',
              '0 10px 24px rgba(0,0,0,0.6)',
              '0 15px 35px rgba(0,0,0,0.7)',
              '0 20px 40px rgba(0,0,0,0.8)',
              '0 25px 50px rgba(0,0,0,0.9)',
              ...Array(17).fill('0 25px 50px rgba(0,0,0,0.9)'),
            ]
          : [
              'none',
              '0 1px 3px rgba(0,0,0,0.05)',
              '0 4px 6px rgba(0,0,0,0.07)',
              '0 5px 15px rgba(0,0,0,0.08)',
              '0 10px 24px rgba(0,0,0,0.1)',
              '0 15px 35px rgba(0,0,0,0.12)',
              '0 20px 40px rgba(0,0,0,0.14)',
              '0 25px 50px rgba(0,0,0,0.15)',
              ...Array(17).fill('0 25px 50px rgba(0,0,0,0.15)'),
            ],
        components: {
          MuiCssBaseline: {
            styleOverrides: {
              '*': {
                boxSizing: 'border-box',
                margin: 0,
                padding: 0,
              },
              html: {
                scrollBehavior: 'smooth',
                WebkitFontSmoothing: 'antialiased',
                MozOsxFontSmoothing: 'grayscale',
              },
              body: {
                overflowX: 'hidden',
                transition: 'background-color 0.3s ease',
              },
              '::-webkit-scrollbar': {
                width: '12px',
                height: '12px',
              },
              '::-webkit-scrollbar-track': {
                background: darkMode ? '#0f172a' : '#f1f5f9',
                borderRadius: '6px',
              },
              '::-webkit-scrollbar-thumb': {
                background: darkMode 
                  ? 'linear-gradient(180deg, #475569 0%, #334155 100%)'
                  : 'linear-gradient(180deg, #cbd5e1 0%, #94a3b8 100%)',
                borderRadius: '6px',
                border: `2px solid ${darkMode ? '#0f172a' : '#f1f5f9'}`,
                '&:hover': {
                  background: darkMode
                    ? 'linear-gradient(180deg, #64748b 0%, #475569 100%)'
                    : 'linear-gradient(180deg, #94a3b8 0%, #64748b 100%)',
                },
              },
            },
          },
          MuiButton: {
            styleOverrides: {
              root: {
                textTransform: 'none',
                fontWeight: 600,
                borderRadius: 12,
                padding: '12px 28px',
                boxShadow: 'none',
                transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                '&:hover': {
                  boxShadow: darkMode
                    ? '0 8px 24px rgba(59,130,246,0.4)'
                    : '0 8px 16px rgba(0,0,0,0.15)',
                  transform: 'translateY(-2px)',
                },
                '&:active': {
                  transform: 'translateY(0)',
                },
              },
              contained: {
                background: darkMode
                  ? 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)'
                  : 'linear-gradient(135deg, #0277bd 0%, #00897b 100%)',
                '&:hover': {
                  background: darkMode
                    ? 'linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)'
                    : 'linear-gradient(135deg, #025a8d 0%, #006d5e 100%)',
                  boxShadow: darkMode
                    ? '0 12px 32px rgba(59,130,246,0.5)'
                    : '0 12px 24px rgba(2,119,189,0.3)',
                },
              },
              outlined: {
                borderWidth: 2,
                borderColor: darkMode ? '#3b82f6' : '#0277bd',
                '&:hover': {
                  borderWidth: 2,
                  backgroundColor: darkMode 
                    ? alpha('#3b82f6', 0.1)
                    : alpha('#0277bd', 0.1),
                },
              },
            },
            variants: [
              {
                props: { variant: 'gradient' },
                style: {
                  background: darkMode
                    ? 'linear-gradient(135deg, #5b21b6 0%, #7c3aed 100%)'
                    : 'linear-gradient(135deg, #0277bd 0%, #00897b 100%)',
                  color: '#ffffff',
                  boxShadow: darkMode
                    ? '0 4px 16px rgba(91,33,182,0.4)'
                    : '0 4px 16px rgba(2,119,189,0.3)',
                  '&:hover': {
                    background: darkMode
                      ? 'linear-gradient(135deg, #6d28d9 0%, #8b5cf6 100%)'
                      : 'linear-gradient(135deg, #025a8d 0%, #006d5e 100%)',
                    boxShadow: darkMode
                      ? '0 12px 32px rgba(91,33,182,0.6)'
                      : '0 12px 24px rgba(2,119,189,0.4)',
                  },
                },
              },
            ],
          },
          MuiCard: {
            styleOverrides: {
              root: {
                borderRadius: 16,
                boxShadow: darkMode 
                  ? '0 8px 32px rgba(0,0,0,0.6)' 
                  : '0 4px 20px rgba(0,0,0,0.08)',
                border: `1px solid ${darkMode ? alpha('#ffffff', 0.05) : alpha('#000000', 0.05)}`,
                transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                background: darkMode
                  ? 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)'
                  : '#ffffff',
                backdropFilter: 'blur(10px)',
                '&:hover': {
                  transform: 'translateY(-6px)',
                  boxShadow: darkMode
                    ? '0 16px 48px rgba(0,0,0,0.8)'
                    : '0 12px 32px rgba(0,0,0,0.12)',
                  border: `1px solid ${darkMode ? alpha('#3b82f6', 0.3) : alpha('#0277bd', 0.2)}`,
                },
              },
            },
          },
          MuiPaper: {
            styleOverrides: {
              root: {
                backgroundImage: 'none',
                backgroundColor: darkMode ? '#132f4c' : '#ffffff',
              },
              elevation1: {
                boxShadow: darkMode 
                  ? '0 2px 8px rgba(0,0,0,0.4)' 
                  : '0 2px 8px rgba(0,0,0,0.06)',
              },
            },
          },
          MuiTextField: {
            styleOverrides: {
              root: {
                '& .MuiOutlinedInput-root': {
                  borderRadius: 12,
                  transition: 'all 0.3s ease',
                  backgroundColor: darkMode ? alpha('#ffffff', 0.05) : 'transparent',
                  '&:hover': {
                    backgroundColor: darkMode ? alpha('#ffffff', 0.08) : alpha('#0277bd', 0.02),
                    '& .MuiOutlinedInput-notchedOutline': {
                      borderColor: darkMode ? '#3b82f6' : '#0277bd',
                      borderWidth: 2,
                    },
                  },
                  '&.Mui-focused': {
                    backgroundColor: darkMode ? alpha('#ffffff', 0.1) : alpha('#0277bd', 0.05),
                    boxShadow: darkMode
                      ? `0 0 0 4px ${alpha('#3b82f6', 0.15)}`
                      : `0 0 0 4px ${alpha('#0277bd', 0.1)}`,
                  },
                },
              },
            },
          },
          MuiChip: {
            styleOverrides: {
              root: {
                fontWeight: 600,
                borderRadius: 10,
                transition: 'all 0.2s ease',
                '&:hover': {
                  transform: 'scale(1.05)',
                },
              },
            },
          },
          MuiTooltip: {
            styleOverrides: {
              tooltip: {
                borderRadius: 10,
                padding: '10px 16px',
                fontSize: '0.875rem',
                fontWeight: 500,
                backgroundColor: darkMode ? '#0f172a' : '#334155',
                boxShadow: '0 8px 24px rgba(0,0,0,0.25)',
                backdropFilter: 'blur(10px)',
                border: `1px solid ${darkMode ? alpha('#ffffff', 0.1) : alpha('#ffffff', 0.2)}`,
              },
              arrow: {
                color: darkMode ? '#0f172a' : '#334155',
              },
            },
          },
        },
      }),
    [darkMode]
  );

  // Enhanced Global Styles with Advanced Animations
  const globalStyles = (
    <GlobalStyles
      styles={{
        '@keyframes fadeIn': {
          from: { opacity: 0, transform: 'translateY(30px)' },
          to: { opacity: 1, transform: 'translateY(0)' },
        },
        '@keyframes slideInLeft': {
          from: { opacity: 0, transform: 'translateX(-40px)' },
          to: { opacity: 1, transform: 'translateX(0)' },
        },
        '@keyframes slideInRight': {
          from: { opacity: 0, transform: 'translateX(40px)' },
          to: { opacity: 1, transform: 'translateX(0)' },
        },
        '@keyframes scaleIn': {
          from: { opacity: 0, transform: 'scale(0.85)' },
          to: { opacity: 1, transform: 'scale(1)' },
        },
        '@keyframes shimmer': {
          '0%': { backgroundPosition: '-1000px 0' },
          '100%': { backgroundPosition: '1000px 0' },
        },
        '@keyframes glow': {
          '0%, 100%': { boxShadow: '0 0 20px rgba(59,130,246,0.3)' },
          '50%': { boxShadow: '0 0 40px rgba(59,130,246,0.6)' },
        },
        '.animate-fade-in': {
          animation: 'fadeIn 0.8s ease-out',
        },
        '.animate-slide-left': {
          animation: 'slideInLeft 0.8s ease-out',
        },
        '.animate-slide-right': {
          animation: 'slideInRight 0.8s ease-out',
        },
        '.animate-scale-in': {
          animation: 'scaleIn 0.6s ease-out',
        },
        '.gradient-text': {
          background: darkMode
            ? 'linear-gradient(135deg, #3b82f6 0%, #10b981 100%)'
            : 'linear-gradient(135deg, #0277bd 0%, #00897b 100%)',
          backgroundClip: 'text',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
        },
        '.glass-effect': {
          background: darkMode 
            ? 'rgba(30,41,59,0.7)'
            : 'rgba(255,255,255,0.7)',
          backdropFilter: 'blur(20px) saturate(180%)',
          border: `1px solid ${darkMode ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`,
        },
      }}
    />
  );

  if (loading) {
    return <PageLoader />;
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {globalStyles}
      <Router>
        <Routes>
          <Route path="/" element={<MainLayout darkMode={darkMode} setDarkMode={setDarkMode} />}>
            <Route index element={<Home />} />
            <Route path="analytics" element={<Analytics />} />
            <Route path="explorer" element={<DataExplorer />} />
            <Route path="chatbot" element={<ChatbotPage />} />
            <Route path="profile" element={<Profile />} />
            <Route path="about" element={<About />} />
            <Route path="contact" element={<Contact />} />
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Router>

      {/* Premium Toast Notifications */}
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: darkMode 
              ? 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)'
              : 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)',
            color: darkMode ? '#f1f5f9' : '#1e293b',
            borderRadius: '16px',
            border: `1px solid ${darkMode ? alpha('#ffffff', 0.1) : alpha('#000000', 0.08)}`,
            boxShadow: darkMode
              ? '0 12px 32px rgba(0,0,0,0.6)'
              : '0 8px 24px rgba(0,0,0,0.12)',
            padding: '16px 24px',
            fontSize: '0.95rem',
            fontWeight: 500,
            backdropFilter: 'blur(20px)',
            maxWidth: '400px',
          },
          success: {
            duration: 3000,
            style: {
              background: darkMode 
                ? `linear-gradient(135deg, ${alpha('#10b981', 0.2)} 0%, ${alpha('#059669', 0.15)} 100%)`
                : `linear-gradient(135deg, ${alpha('#10b981', 0.1)} 0%, ${alpha('#ecfdf5', 1)} 100%)`,
              borderColor: '#10b981',
              boxShadow: '0 8px 24px rgba(16,185,129,0.3)',
            },
            iconTheme: {
              primary: '#10b981',
              secondary: darkMode ? '#1e293b' : '#ffffff',
            },
          },
          error: {
            duration: 5000,
            style: {
              background: darkMode 
                ? `linear-gradient(135deg, ${alpha('#ef4444', 0.2)} 0%, ${alpha('#dc2626', 0.15)} 100%)`
                : `linear-gradient(135deg, ${alpha('#ef4444', 0.1)} 0%, ${alpha('#fef2f2', 1)} 100%)`,
              borderColor: '#ef4444',
              boxShadow: '0 8px 24px rgba(239,68,68,0.3)',
            },
            iconTheme: {
              primary: '#ef4444',
              secondary: darkMode ? '#1e293b' : '#ffffff',
            },
          },
          loading: {
            style: {
              background: darkMode 
                ? `linear-gradient(135deg, ${alpha('#3b82f6', 0.2)} 0%, ${alpha('#2563eb', 0.15)} 100%)`
                : `linear-gradient(135deg, ${alpha('#3b82f6', 0.1)} 0%, ${alpha('#eff6ff', 1)} 100%)`,
              borderColor: '#3b82f6',
              boxShadow: '0 8px 24px rgba(59,130,246,0.3)',
            },
            iconTheme: {
              primary: '#3b82f6',
              secondary: darkMode ? '#1e293b' : '#ffffff',
            },
          },
        }}
      />
    </ThemeProvider>
  );
}

export default App;