import React, { useState, useEffect } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { Box, Fab, Zoom, Tooltip, alpha, IconButton, Snackbar, Alert } from '@mui/material';
import Navbar from '../components/Navbar/Navbar';
import Sidebar from '../components/Sidebar/Sidebar';
import Footer from '../components/Footer/Footer';
import KeyboardArrowUpIcon from '@mui/icons-material/KeyboardArrowUp';
import ChatIcon from '@mui/icons-material/Chat';
import LightModeIcon from '@mui/icons-material/LightMode';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';

const MainLayout = ({ darkMode, setDarkMode }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [showScrollTop, setShowScrollTop] = useState(false);
  const [showWelcome, setShowWelcome] = useState(true);
  const location = useLocation();

  // Scroll to top on route change
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [location.pathname]);

  // Handle scroll for "back to top" button
  useEffect(() => {
    const handleScroll = () => {
      setShowScrollTop(window.scrollY > 400);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleSidebarToggle = () => {
    setSidebarOpen(!sidebarOpen);
  };

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleChatClick = () => {
    window.location.href = '/chatbot';
  };

  const handleHelpClick = () => {
    window.location.href = '/about';
  };

  return (
    <Box 
      sx={{ 
        display: 'flex', 
        flexDirection: 'column', 
        minHeight: '100vh',
        position: 'relative',
        transition: 'background-color 0.3s ease, color 0.3s ease',
      }}
    >
      {/* Navbar */}
      <Navbar 
        onMenuClick={handleSidebarToggle} 
        darkMode={darkMode}
        setDarkMode={setDarkMode}
      />

      {/* Main Content Area */}
      <Box sx={{ display: 'flex', flex: 1, position: 'relative' }}>
        {/* Sidebar */}
        <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

        {/* Main Content */}
        <Box 
          component='main' 
          sx={{ 
            flexGrow: 1,
            position: 'relative',
            minHeight: 'calc(100vh - 64px)',
            backgroundColor: darkMode ? '#0a1929' : '#f8f9fa',
            transition: 'all 0.3s ease',
            // Animated background pattern
            '&::before': {
              content: '""',
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundImage: darkMode
                ? `radial-gradient(circle at 20% 50%, ${alpha('#0277bd', 0.03)} 0%, transparent 50%),
                   radial-gradient(circle at 80% 80%, ${alpha('#00897b', 0.03)} 0%, transparent 50%)`
                : `radial-gradient(circle at 20% 50%, ${alpha('#0277bd', 0.02)} 0%, transparent 50%),
                   radial-gradient(circle at 80% 80%, ${alpha('#00897b', 0.02)} 0%, transparent 50%)`,
              pointerEvents: 'none',
              zIndex: 0,
            },
          }}
        >
          {/* Content Container */}
          <Box 
            sx={{ 
              position: 'relative', 
              zIndex: 1,
              p: { xs: 2, sm: 3, md: 4 },
              // Page transition animation
              animation: 'fadeInUp 0.5s ease-out',
              '@keyframes fadeInUp': {
                '0%': {
                  opacity: 0,
                  transform: 'translateY(20px)',
                },
                '100%': {
                  opacity: 1,
                  transform: 'translateY(0)',
                },
              },
            }}
          >
            <Outlet />
          </Box>
        </Box>
      </Box>

      {/* Footer */}
      <Footer />

      {/* Floating Action Buttons */}
      <Box
        sx={{
          position: 'fixed',
          bottom: { xs: 16, md: 24 },
          right: { xs: 16, md: 24 },
          zIndex: 1000,
          display: 'flex',
          flexDirection: 'column',
          gap: 2,
        }}
      >
        {/* Back to Top Button */}
        <Zoom in={showScrollTop}>
          <Tooltip title="Back to Top" placement="left" arrow>
            <Fab
              size="medium"
              onClick={scrollToTop}
              sx={{
                background: darkMode 
                  ? 'linear-gradient(135deg, #1e293b 0%, #334155 100%)'
                  : 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)',
                color: darkMode ? '#f1f5f9' : '#1e293b',
                boxShadow: darkMode
                  ? '0 4px 16px rgba(0,0,0,0.4)'
                  : '0 4px 16px rgba(0,0,0,0.1)',
                border: `1px solid ${darkMode ? alpha('#fff', 0.1) : alpha('#000', 0.08)}`,
                transition: 'all 0.3s ease',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: darkMode
                    ? '0 8px 24px rgba(0,0,0,0.5)'
                    : '0 8px 24px rgba(0,0,0,0.15)',
                  background: darkMode
                    ? 'linear-gradient(135deg, #334155 0%, #475569 100%)'
                    : 'linear-gradient(135deg, #f8f9fa 0%, #e2e8f0 100%)',
                },
              }}
            >
              <KeyboardArrowUpIcon />
            </Fab>
          </Tooltip>
        </Zoom>

        {/* Theme Toggle Button */}
        <Tooltip title={darkMode ? "Light Mode" : "Dark Mode"} placement="left" arrow>
          <Fab
            size="medium"
            onClick={() => setDarkMode(!darkMode)}
            sx={{
              background: darkMode
                ? 'linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%)'
                : 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
              color: 'white',
              boxShadow: darkMode
                ? '0 4px 16px rgba(251, 191, 36, 0.3)'
                : '0 4px 16px rgba(30, 41, 59, 0.3)',
              transition: 'all 0.3s ease',
              '&:hover': {
                transform: 'translateY(-4px) rotate(180deg)',
                boxShadow: darkMode
                  ? '0 8px 24px rgba(251, 191, 36, 0.4)'
                  : '0 8px 24px rgba(30, 41, 59, 0.4)',
              },
            }}
          >
            {darkMode ? <LightModeIcon /> : <DarkModeIcon />}
          </Fab>
        </Tooltip>

        {/* AI Chat Button */}
        <Tooltip title="AI Assistant" placement="left" arrow>
          <Fab
            color="primary"
            size="large"
            onClick={handleChatClick}
            sx={{
              background: 'linear-gradient(135deg, #0277bd 0%, #00897b 100%)',
              boxShadow: '0 4px 16px rgba(2, 119, 189, 0.3)',
              transition: 'all 0.3s ease',
              animation: 'pulse 2s ease-in-out infinite',
              '&:hover': {
                transform: 'translateY(-4px) scale(1.05)',
                boxShadow: '0 8px 24px rgba(2, 119, 189, 0.4)',
                animation: 'none',
              },
              '@keyframes pulse': {
                '0%, 100%': {
                  boxShadow: '0 4px 16px rgba(2, 119, 189, 0.3)',
                },
                '50%': {
                  boxShadow: '0 4px 20px rgba(2, 119, 189, 0.5)',
                },
              },
            }}
          >
            <ChatIcon sx={{ fontSize: 28 }} />
          </Fab>
        </Tooltip>

        {/* Help Button */}
        <Tooltip title="Help & Support" placement="left" arrow>
          <Fab
            size="small"
            onClick={handleHelpClick}
            sx={{
              background: darkMode
                ? 'linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%)'
                : 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)',
              color: 'white',
              boxShadow: '0 4px 16px rgba(124, 58, 237, 0.3)',
              transition: 'all 0.3s ease',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: '0 8px 24px rgba(124, 58, 237, 0.4)',
              },
            }}
          >
            <HelpOutlineIcon fontSize="small" />
          </Fab>
        </Tooltip>
      </Box>

      {/* Welcome Snackbar */}
      <Snackbar
        open={showWelcome}
        autoHideDuration={6000}
        onClose={() => setShowWelcome(false)}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert
          onClose={() => setShowWelcome(false)}
          severity="info"
          variant="filled"
          sx={{
            background: 'linear-gradient(135deg, #0277bd 0%, #00897b 100%)',
            color: 'white',
            fontWeight: 600,
            boxShadow: '0 8px 24px rgba(2, 119, 189, 0.3)',
            '& .MuiAlert-icon': {
              color: 'white',
            },
          }}
        >
          Welcome to MeenaSetu AI - Your Intelligent Aquatic Expert Platform! 🐠
        </Alert>
      </Snackbar>

      {/* Decorative Elements */}
      <Box
        sx={{
          position: 'fixed',
          top: -100,
          right: -100,
          width: 300,
          height: 300,
          borderRadius: '50%',
          background: darkMode
            ? `radial-gradient(circle, ${alpha('#0277bd', 0.05)} 0%, transparent 70%)`
            : `radial-gradient(circle, ${alpha('#0277bd', 0.03)} 0%, transparent 70%)`,
          pointerEvents: 'none',
          zIndex: 0,
          animation: 'float 6s ease-in-out infinite',
          '@keyframes float': {
            '0%, 100%': {
              transform: 'translate(0, 0)',
            },
            '50%': {
              transform: 'translate(-20px, 20px)',
            },
          },
        }}
      />

      <Box
        sx={{
          position: 'fixed',
          bottom: -100,
          left: -100,
          width: 300,
          height: 300,
          borderRadius: '50%',
          background: darkMode
            ? `radial-gradient(circle, ${alpha('#00897b', 0.05)} 0%, transparent 70%)`
            : `radial-gradient(circle, ${alpha('#00897b', 0.03)} 0%, transparent 70%)`,
          pointerEvents: 'none',
          zIndex: 0,
          animation: 'float 8s ease-in-out infinite reverse',
        }}
      />
    </Box>
  );
};

export default MainLayout;