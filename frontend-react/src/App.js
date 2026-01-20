import React, { useState, useEffect, useMemo, Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline, alpha, GlobalStyles, Box, LinearProgress } from '@mui/material';
import { Toaster } from 'react-hot-toast';
import { AnimatePresence, motion } from 'framer-motion';

// Eager load main layout
import MainLayout from './layouts/MainLayout';

// Lazy load pages for better performance
const Home = lazy(() => import('./pages/Home/Home'));
const Analytics = lazy(() => import('./pages/Analytics/Analytics'));
const DataExplorer = lazy(() => import('./pages/DataExplorer/DataExplorer'));
const ChatbotPage = lazy(() => import('./pages/ChatbotPage/ChatbotPage'));
const Profile = lazy(() => import('./pages/Profile/Profile'));
const About = lazy(() => import('./pages/About/About'));
const Contact = lazy(() => import('./pages/Contact/Contact'));

// Elegant Loading Component
const PageLoader = () => (
  <Box
    sx={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #0277bd 0%, #00897b 100%)',
      zIndex: 9999,
    }}
  >
    <Box
      sx={{
        width: 80,
        height: 80,
        mb: 3,
        position: 'relative',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          border: '4px solid rgba(255,255,255,0.2)',
          borderRadius: '50%',
        },
        '&::after': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          border: '4px solid transparent',
          borderTopColor: '#fff',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
        },
        '@keyframes spin': {
          '0%': { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(360deg)' },
        },
      }}
    />
    <Box sx={{ color: 'white', fontSize: '1.25rem', fontWeight: 600, mb: 1 }}>
      MeenaSetu AI
    </Box>
    <Box sx={{ color: 'rgba(255,255,255,0.8)', fontSize: '0.875rem' }}>
      Loading Aquatic Intelligence...
    </Box>
  </Box>
);

// Suspense Fallback with Progress Bar
const SuspenseFallback = () => (
  <Box sx={{ width: '100%', position: 'fixed', top: 0, left: 0, zIndex: 9999 }}>
    <LinearProgress
      sx={{
        height: 3,
        backgroundColor: alpha('#0277bd', 0.1),
        '& .MuiLinearProgress-bar': {
          background: 'linear-gradient(90deg, #0277bd 0%, #00897b 100%)',
        },
      }}
    />
  </Box>
);

// Animated Route Wrapper
const AnimatedRoute = ({ children }) => {
  const location = useLocation();
  
  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={location.pathname}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        transition={{ duration: 0.3, ease: 'easeInOut' }}
      >
        {children}
      </motion.div>
    </AnimatePresence>
  );
};

function App() {
  const [loading, setLoading] = useState(true);
  const [darkMode, setDarkMode] = useState(() => {
    const savedMode = localStorage.getItem('meenasetu_theme_mode');
    return savedMode === 'dark';
  });

  // Initial loading effect
  useEffect(() => {
    const timer = setTimeout(() => setLoading(false), 1800);
    return () => clearTimeout(timer);
  }, []);

  // Persist theme preference
  useEffect(() => {
    localStorage.setItem('meenasetu_theme_mode', darkMode ? 'dark' : 'light');
    document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  // Dynamic Enhanced Theme
  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          mode: darkMode ? 'dark' : 'light',
          primary: {
            main: '#0277bd',
            light: '#58a5f0',
            dark: '#004c8c',
            contrastText: '#ffffff',
          },
          secondary: {
            main: '#00897b',
            light: '#4ebaaa',
            dark: '#005b4f',
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
          text: {
            primary: darkMode ? '#f1f5f9' : '#1e293b',
            secondary: darkMode ? '#94a3b8' : '#64748b',
          },
        },
        typography: {
          fontFamily: '"Inter", "SF Pro Display", -apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", sans-serif',
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
            lineHeight: 1.4,
          },
          h5: {
            fontSize: '1.25rem',
            fontWeight: 600,
            lineHeight: 1.4,
          },
          h6: {
            fontSize: '1rem',
            fontWeight: 600,
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
        shadows: [
          'none',
          darkMode ? '0 1px 3px rgba(0,0,0,0.4)' : '0 1px 3px rgba(0,0,0,0.05)',
          darkMode ? '0 4px 6px rgba(0,0,0,0.5)' : '0 4px 6px rgba(0,0,0,0.07)',
          darkMode ? '0 5px 15px rgba(0,0,0,0.6)' : '0 5px 15px rgba(0,0,0,0.08)',
          darkMode ? '0 10px 24px rgba(0,0,0,0.7)' : '0 10px 24px rgba(0,0,0,0.1)',
          darkMode ? '0 15px 35px rgba(0,0,0,0.8)' : '0 15px 35px rgba(0,0,0,0.12)',
          ...Array(19).fill(darkMode ? '0 20px 40px rgba(0,0,0,0.9)' : '0 20px 40px rgba(0,0,0,0.14)'),
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
                transition: 'background-color 0.3s ease, color 0.3s ease',
              },
              '::-webkit-scrollbar': {
                width: '12px',
                height: '12px',
              },
              '::-webkit-scrollbar-track': {
                background: darkMode ? '#1e293b' : '#f1f5f9',
                borderRadius: '10px',
              },
              '::-webkit-scrollbar-thumb': {
                background: darkMode 
                  ? 'linear-gradient(180deg, #475569 0%, #334155 100%)'
                  : 'linear-gradient(180deg, #cbd5e1 0%, #94a3b8 100%)',
                borderRadius: '10px',
                border: `2px solid ${darkMode ? '#1e293b' : '#f1f5f9'}`,
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
                borderRadius: 10,
                padding: '10px 24px',
                boxShadow: 'none',
                transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                '&:hover': {
                  transform: 'translateY(-2px)',
                  boxShadow: darkMode 
                    ? '0 8px 16px rgba(0,0,0,0.4)' 
                    : '0 8px 16px rgba(0,0,0,0.15)',
                },
                '&:active': {
                  transform: 'translateY(0)',
                },
              },
              contained: {
                '&:hover': {
                  boxShadow: darkMode
                    ? '0 12px 24px rgba(0,0,0,0.5)'
                    : '0 12px 24px rgba(0,0,0,0.2)',
                },
              },
              outlined: {
                borderWidth: 2,
                '&:hover': {
                  borderWidth: 2,
                },
              },
            },
          },
          MuiCard: {
            styleOverrides: {
              root: {
                borderRadius: 16,
                boxShadow: darkMode 
                  ? '0 4px 20px rgba(0,0,0,0.5)' 
                  : '0 4px 20px rgba(0,0,0,0.08)',
                border: `1px solid ${darkMode ? alpha('#ffffff', 0.08) : alpha('#000000', 0.05)}`,
                transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                backgroundImage: 'none',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: darkMode
                    ? '0 12px 32px rgba(0,0,0,0.6)'
                    : '0 12px 32px rgba(0,0,0,0.12)',
                },
              },
            },
          },
          MuiPaper: {
            styleOverrides: {
              root: {
                backgroundImage: 'none',
              },
            },
          },
          MuiTextField: {
            styleOverrides: {
              root: {
                '& .MuiOutlinedInput-root': {
                  borderRadius: 10,
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    '& .MuiOutlinedInput-notchedOutline': {
                      borderColor: '#0277bd',
                    },
                  },
                  '&.Mui-focused': {
                    boxShadow: `0 0 0 3px ${alpha('#0277bd', 0.15)}`,
                  },
                },
              },
            },
          },
          MuiChip: {
            styleOverrides: {
              root: {
                fontWeight: 600,
                borderRadius: 8,
              },
            },
          },
          MuiTooltip: {
            styleOverrides: {
              tooltip: {
                borderRadius: 8,
                padding: '8px 12px',
                fontSize: '0.875rem',
                fontWeight: 500,
                backgroundColor: darkMode ? '#1e293b' : '#334155',
                boxShadow: darkMode
                  ? '0 4px 12px rgba(0,0,0,0.4)'
                  : '0 4px 12px rgba(0,0,0,0.15)',
              },
            },
          },
        },
      }),
    [darkMode]
  );

  // Global Animation Styles
  const globalStyles = (
    <GlobalStyles
      styles={{
        '@keyframes fadeIn': {
          from: { opacity: 0, transform: 'translateY(20px)' },
          to: { opacity: 1, transform: 'translateY(0)' },
        },
        '@keyframes slideInLeft': {
          from: { opacity: 0, transform: 'translateX(-30px)' },
          to: { opacity: 1, transform: 'translateX(0)' },
        },
        '@keyframes slideInRight': {
          from: { opacity: 0, transform: 'translateX(30px)' },
          to: { opacity: 1, transform: 'translateX(0)' },
        },
        '@keyframes scaleIn': {
          from: { opacity: 0, transform: 'scale(0.9)' },
          to: { opacity: 1, transform: 'scale(1)' },
        },
        '@keyframes float': {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        '.animate-fade-in': {
          animation: 'fadeIn 0.6s ease-out',
        },
        '.animate-slide-left': {
          animation: 'slideInLeft 0.6s ease-out',
        },
        '.animate-slide-right': {
          animation: 'slideInRight 0.6s ease-out',
        },
        '.animate-scale-in': {
          animation: 'scaleIn 0.5s ease-out',
        },
        '.animate-float': {
          animation: 'float 3s ease-in-out infinite',
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
        <Suspense fallback={<SuspenseFallback />}>
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
        </Suspense>
      </Router>

      {/* Premium Toast Notifications */}
      <Toaster
        position="top-right"
        gutter={12}
        toastOptions={{
          duration: 4000,
          style: {
            background: darkMode ? '#1e293b' : '#ffffff',
            color: darkMode ? '#f1f5f9' : '#1e293b',
            borderRadius: '12px',
            border: `1px solid ${darkMode ? alpha('#ffffff', 0.1) : alpha('#000000', 0.1)}`,
            boxShadow: darkMode
              ? '0 10px 30px rgba(0,0,0,0.5)'
              : '0 10px 30px rgba(0,0,0,0.15)',
            padding: '16px 20px',
            fontSize: '0.95rem',
            fontWeight: 500,
            backdropFilter: 'blur(10px)',
            maxWidth: '400px',
          },
          success: {
            duration: 3000,
            style: {
              background: darkMode 
                ? alpha('#10b981', 0.15) 
                : alpha('#10b981', 0.1),
              borderColor: '#10b981',
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
                ? alpha('#ef4444', 0.15) 
                : alpha('#ef4444', 0.1),
              borderColor: '#ef4444',
            },
            iconTheme: {
              primary: '#ef4444',
              secondary: darkMode ? '#1e293b' : '#ffffff',
            },
          },
          loading: {
            style: {
              background: darkMode 
                ? alpha('#3b82f6', 0.15) 
                : alpha('#3b82f6', 0.1),
              borderColor: '#3b82f6',
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