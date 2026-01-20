import React, { useState } from 'react';
import {
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemButton,
  Divider,
  Box,
  Typography,
  Avatar,
  Chip,
  alpha,
  IconButton,
  LinearProgress,
  Paper,
} from '@mui/material';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Home as HomeIcon,
  BarChart as BarChartIcon,
  Storage as StorageIcon,
  Chat as ChatIcon,
  Person as PersonIcon,
  Info as InfoIcon,
  Mail as MailIcon,
  Waves as WavesIcon,
  TrendingUp,
  AutoAwesome,
  Close as CloseIcon,
  Favorite,
  Settings,
  EmojiEvents,
} from '@mui/icons-material';

const Sidebar = ({ open, onClose }) => {
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems = [
    {
      text: 'Home',
      icon: <HomeIcon />,
      path: '/',
      badge: null,
      gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      description: 'Dashboard overview',
    },
    {
      text: 'Analytics',
      icon: <BarChartIcon />,
      path: '/analytics',
      badge: 'New',
      gradient: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
      description: 'Data insights & trends',
    },
    {
      text: 'Data Management',
      icon: <StorageIcon />,
      path: '/data',
      badge: null,
      gradient: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
      description: 'Upload & manage data',
    },
    {
      text: 'AI Chatbot',
      icon: <ChatIcon />,
      path: '/chatbot',
      badge: 'AI',
      gradient: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
      description: 'Smart conversations',
    },
  ];

  const secondaryItems = [
    { text: 'Profile', icon: <PersonIcon />, path: '/profile' },
    { text: 'Settings', icon: <Settings />, path: '/settings' },
    { text: 'About', icon: <InfoIcon />, path: '/about' },
    { text: 'Contact', icon: <MailIcon />, path: '/contact' },
  ];

  const handleNavigate = (path) => {
    navigate(path);
    if (onClose) {
      onClose();
    }
  };

  const handleUpgradeClick = () => {
    navigate('/upgrade');
    if (onClose) {
      onClose();
    }
  };

  const handleFeatureClick = () => {
    navigate('/analytics');
    if (onClose) {
      onClose();
    }
  };

  return (
    <Drawer
      anchor="left"
      open={open}
      onClose={onClose}
      PaperProps={{
        sx: {
          width: 320,
          background: 'linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%)',
          borderRight: 'none',
          boxShadow: '4px 0 24px rgba(0,0,0,0.12)',
        },
      }}
    >
      <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        {/* Header Section */}
        <Box
          sx={{
            p: 3,
            pt: 3,
            background: 'linear-gradient(135deg, #0277bd 0%, #00897b 100%)',
            color: 'white',
            position: 'relative',
            overflow: 'hidden',
            '&::before': {
              content: '""',
              position: 'absolute',
              top: -50,
              right: -50,
              width: 200,
              height: 200,
              background: 'radial-gradient(circle, rgba(255,255,255,0.2) 0%, transparent 70%)',
              animation: 'pulse 4s ease-in-out infinite',
            },
            '&::after': {
              content: '""',
              position: 'absolute',
              bottom: -30,
              left: -30,
              width: 150,
              height: 150,
              background: 'radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%)',
              animation: 'pulse 6s ease-in-out infinite',
            },
            '@keyframes pulse': {
              '0%, 100%': { transform: 'scale(1)', opacity: 0.5 },
              '50%': { transform: 'scale(1.1)', opacity: 0.8 },
            },
          }}
        >
          {/* Close Button */}
          <IconButton
            onClick={onClose}
            sx={{
              position: 'absolute',
              top: 12,
              right: 12,
              color: 'white',
              bgcolor: 'rgba(255, 255, 255, 0.15)',
              backdropFilter: 'blur(10px)',
              '&:hover': {
                bgcolor: 'rgba(255, 255, 255, 0.25)',
                transform: 'rotate(90deg)',
              },
              transition: 'all 0.3s ease-in-out',
              zIndex: 1,
            }}
          >
            <CloseIcon fontSize="small" />
          </IconButton>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, position: 'relative', mb: 2.5 }}>
            <Avatar
              sx={{
                width: 56,
                height: 56,
                background: 'rgba(255, 255, 255, 0.2)',
                backdropFilter: 'blur(10px)',
                border: '3px solid rgba(255, 255, 255, 0.3)',
                boxShadow: '0 8px 24px rgba(0,0,0,0.2)',
              }}
            >
              <WavesIcon sx={{ fontSize: 32 }} />
            </Avatar>
            <Box>
              <Typography variant="h5" fontWeight={700} letterSpacing={-0.5}>
                MeenaSetu
              </Typography>
              <Typography variant="caption" sx={{ opacity: 0.95, fontSize: '0.8rem' }}>
                Aquatic Intelligence
              </Typography>
            </Box>
          </Box>

          {/* Stats Card */}
          <Paper
            elevation={0}
            sx={{
              p: 2,
              background: 'rgba(255, 255, 255, 0.15)',
              backdropFilter: 'blur(10px)',
              borderRadius: 2,
              border: '1px solid rgba(255, 255, 255, 0.25)',
              position: 'relative',
            }}
          >
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1.5 }}>
              <Box>
                <Typography variant="h6" fontWeight={700} sx={{ lineHeight: 1 }}>
                  87%
                </Typography>
                <Typography variant="caption" sx={{ opacity: 0.9, fontSize: '0.75rem' }}>
                  Analytics Score
                </Typography>
              </Box>
              <Box sx={{ textAlign: 'right' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <TrendingUp sx={{ fontSize: 16, color: '#4caf50' }} />
                  <Typography variant="body2" fontWeight={600} sx={{ color: '#4caf50' }}>
                    +12%
                  </Typography>
                </Box>
                <Typography variant="caption" sx={{ opacity: 0.9, fontSize: '0.75rem' }}>
                  This week
                </Typography>
              </Box>
            </Box>
            <LinearProgress
              variant="determinate"
              value={87}
              sx={{
                height: 6,
                borderRadius: 3,
                bgcolor: 'rgba(255, 255, 255, 0.2)',
                '& .MuiLinearProgress-bar': {
                  background: 'linear-gradient(90deg, #4caf50 0%, #8bc34a 100%)',
                  borderRadius: 3,
                },
              }}
            />
          </Paper>
        </Box>

        {/* Main Navigation */}
        <Box sx={{ p: 2.5, flex: 1, overflowY: 'auto' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1.5 }}>
            <Typography
              variant="caption"
              sx={{
                px: 1,
                fontWeight: 700,
                color: 'text.secondary',
                textTransform: 'uppercase',
                letterSpacing: 1.2,
                fontSize: '0.7rem',
              }}
            >
              Navigation
            </Typography>
            <Chip
              label="4"
              size="small"
              sx={{
                height: 20,
                fontSize: '0.7rem',
                bgcolor: alpha('#0277bd', 0.1),
                color: '#0277bd',
                fontWeight: 600,
              }}
            />
          </Box>

          <List sx={{ px: 0 }}>
            {menuItems.map((item) => {
              const isActive = location.pathname === item.path;

              return (
                <ListItem key={item.text} disablePadding sx={{ mb: 1 }}>
                  <ListItemButton
                    onClick={() => handleNavigate(item.path)}
                    sx={{
                      borderRadius: 2.5,
                      py: 1.8,
                      px: 2,
                      position: 'relative',
                      overflow: 'hidden',
                      backgroundColor: isActive ? alpha('#0277bd', 0.08) : 'transparent',
                      boxShadow: isActive ? `0 4px 12px ${alpha('#0277bd', 0.15)}` : 'none',
                      '&:hover': {
                        backgroundColor: isActive ? alpha('#0277bd', 0.12) : alpha('#0277bd', 0.06),
                        transform: 'translateX(6px)',
                        boxShadow: `0 4px 16px ${alpha('#0277bd', 0.12)}`,
                      },
                      '&::before': isActive
                        ? {
                            content: '""',
                            position: 'absolute',
                            left: 0,
                            top: '15%',
                            bottom: '15%',
                            width: 5,
                            borderRadius: '0 6px 6px 0',
                            background: item.gradient,
                            boxShadow: `2px 0 8px ${alpha('#0277bd', 0.3)}`,
                          }
                        : {},
                      transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
                    }}
                  >
                    <ListItemIcon sx={{ minWidth: 48 }}>
                      <Box
                        sx={{
                          background: isActive ? item.gradient : alpha('#0277bd', 0.08),
                          borderRadius: '12px',
                          p: 1,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          color: isActive ? 'white' : '#0277bd',
                          transition: 'all 0.3s ease-in-out',
                          boxShadow: isActive ? `0 4px 12px ${alpha('#0277bd', 0.3)}` : 'none',
                          transform: isActive ? 'scale(1.05)' : 'scale(1)',
                        }}
                      >
                        {item.icon}
                      </Box>
                    </ListItemIcon>
                    <ListItemText
                      primary={item.text}
                      secondary={item.description}
                      primaryTypographyProps={{
                        fontWeight: isActive ? 700 : 600,
                        fontSize: '0.95rem',
                        color: isActive ? '#0277bd' : 'text.primary',
                      }}
                      secondaryTypographyProps={{
                        fontSize: '0.75rem',
                        sx: {
                          mt: 0.3,
                          opacity: isActive ? 0.9 : 0.7,
                        },
                      }}
                    />
                    {item.badge && (
                      <Chip
                        label={item.badge}
                        size="small"
                        sx={{
                          height: 24,
                          fontSize: '0.7rem',
                          fontWeight: 700,
                          background: item.gradient,
                          color: 'white',
                          border: 'none',
                          boxShadow: `0 2px 8px ${alpha('#0277bd', 0.3)}`,
                        }}
                      />
                    )}
                  </ListItemButton>
                </ListItem>
              );
            })}
          </List>

          {/* Feature Highlight */}
          <Paper
            elevation={0}
            sx={{
              mt: 2,
              p: 2,
              borderRadius: 2.5,
              background: 'linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%)',
              border: '1px solid rgba(2, 119, 189, 0.1)',
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 1 }}>
              <AutoAwesome sx={{ color: '#0277bd', fontSize: 20 }} />
              <Typography variant="body2" fontWeight={700} color="#0277bd">
                AI Feature
              </Typography>
            </Box>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              Try our new predictive analytics powered by machine learning
            </Typography>
            <Box
              onClick={handleFeatureClick}
              sx={{
                bgcolor: '#0277bd',
                color: 'white',
                py: 0.8,
                px: 1.5,
                borderRadius: 1.5,
                cursor: 'pointer',
                fontWeight: 600,
                fontSize: '0.8rem',
                textAlign: 'center',
                '&:hover': {
                  bgcolor: '#01579b',
                  transform: 'translateY(-2px)',
                  boxShadow: `0 4px 12px ${alpha('#0277bd', 0.4)}`,
                },
                transition: 'all 0.2s',
              }}
            >
              Explore Now
            </Box>
          </Paper>

          {/* Divider with Icon */}
          <Box sx={{ display: 'flex', alignItems: 'center', my: 3 }}>
            <Divider sx={{ flex: 1 }} />
            <Favorite sx={{ mx: 2, fontSize: 16, color: 'text.secondary', opacity: 0.5 }} />
            <Divider sx={{ flex: 1 }} />
          </Box>

          {/* Secondary Navigation */}
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1.5 }}>
            <Typography
              variant="caption"
              sx={{
                px: 1,
                fontWeight: 700,
                color: 'text.secondary',
                textTransform: 'uppercase',
                letterSpacing: 1.2,
                fontSize: '0.7rem',
              }}
            >
              More Options
            </Typography>
          </Box>

          <List sx={{ px: 0 }}>
            {secondaryItems.map((item) => {
              const isActive = location.pathname === item.path;

              return (
                <ListItem key={item.text} disablePadding sx={{ mb: 0.8 }}>
                  <ListItemButton
                    onClick={() => handleNavigate(item.path)}
                    sx={{
                      borderRadius: 2,
                      py: 1.3,
                      px: 2,
                      backgroundColor: isActive ? alpha('#0277bd', 0.08) : 'transparent',
                      '&:hover': {
                        backgroundColor: isActive ? alpha('#0277bd', 0.12) : alpha('#0277bd', 0.06),
                        transform: 'translateX(6px)',
                      },
                      transition: 'all 0.2s ease-in-out',
                    }}
                  >
                    <ListItemIcon
                      sx={{
                        minWidth: 42,
                        color: isActive ? '#0277bd' : 'text.secondary',
                      }}
                    >
                      {item.icon}
                    </ListItemIcon>
                    <ListItemText
                      primary={item.text}
                      primaryTypographyProps={{
                        fontWeight: isActive ? 600 : 500,
                        fontSize: '0.9rem',
                      }}
                    />
                  </ListItemButton>
                </ListItem>
              );
            })}
          </List>
        </Box>

        {/* Premium Footer */}
        <Box sx={{ p: 2.5 }}>
          <Paper
            elevation={0}
            sx={{
              p: 2.5,
              borderRadius: 2.5,
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              color: 'white',
              textAlign: 'center',
              position: 'relative',
              overflow: 'hidden',
              '&::before': {
                content: '""',
                position: 'absolute',
                top: -20,
                right: -20,
                width: 100,
                height: 100,
                background: 'radial-gradient(circle, rgba(255,255,255,0.2) 0%, transparent 70%)',
              },
            }}
          >
            <Box sx={{ position: 'relative', zIndex: 1 }}>
              <EmojiEvents sx={{ fontSize: 36, mb: 1, opacity: 0.95 }} />
              <Typography variant="body2" fontWeight={700} gutterBottom sx={{ fontSize: '0.95rem' }}>
                Upgrade to Premium
              </Typography>
              <Typography variant="caption" sx={{ opacity: 0.95, display: 'block', mb: 2, fontSize: '0.8rem' }}>
                Unlock advanced AI insights & analytics
              </Typography>
              <Box
                onClick={handleUpgradeClick}
                sx={{
                  bgcolor: 'rgba(255,255,255,0.25)',
                  backdropFilter: 'blur(10px)',
                  py: 1.2,
                  borderRadius: 1.5,
                  cursor: 'pointer',
                  fontWeight: 700,
                  fontSize: '0.85rem',
                  border: '1px solid rgba(255,255,255,0.3)',
                  '&:hover': {
                    bgcolor: 'rgba(255,255,255,0.35)',
                    transform: 'translateY(-2px)',
                    boxShadow: '0 6px 20px rgba(0,0,0,0.2)',
                  },
                  transition: 'all 0.2s',
                }}
              >
                Upgrade Now →
              </Box>
            </Box>
          </Paper>
        </Box>
      </Box>
    </Drawer>
  );
};

export default Sidebar;