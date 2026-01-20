import React, { useState } from 'react';
import {
  Box,
  Typography,
  Container,
  Grid,
  Link,
  IconButton,
  Divider,
  Stack,
  Chip,
  alpha,
  TextField,
  Button,
  Snackbar,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import TwitterIcon from '@mui/icons-material/Twitter';
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import GitHubIcon from '@mui/icons-material/GitHub';
import EmailIcon from '@mui/icons-material/Email';
import WavesIcon from '@mui/icons-material/Waves';
import FiberManualRecordIcon from '@mui/icons-material/FiberManualRecord';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import SecurityIcon from '@mui/icons-material/Security';
import SpeedIcon from '@mui/icons-material/Speed';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import PhoneIcon from '@mui/icons-material/Phone';
import LocationOnIcon from '@mui/icons-material/LocationOn';
import SendIcon from '@mui/icons-material/Send';
import DashboardIcon from '@mui/icons-material/Dashboard';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import DataObjectIcon from '@mui/icons-material/DataObject';
import PsychologyIcon from '@mui/icons-material/Psychology';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import PersonIcon from '@mui/icons-material/Person';
import DescriptionIcon from '@mui/icons-material/Description';
import CodeIcon from '@mui/icons-material/Code';
import SchoolIcon from '@mui/icons-material/School';
import YouTubeIcon from '@mui/icons-material/YouTube';
import ThermostatIcon from '@mui/icons-material/Thermostat';
import OpacityIcon from '@mui/icons-material/Opacity';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import NatureIcon from '@mui/icons-material/Nature';

const Footer = () => {
  const currentYear = new Date().getFullYear();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const [email, setEmail] = useState('');
  const [snackbar, setSnackbar] = useState({ open: false, message: '' });

  const handleNewsletterSubmit = (e) => {
    e.preventDefault();
    if (email) {
      setSnackbar({ open: true, message: 'Thank you for subscribing to our newsletter!' });
      setEmail('');
    }
  };

  const quickLinks = [
    { name: 'Dashboard', icon: <DashboardIcon sx={{ fontSize: 18 }} />, href: '/dashboard' },
    { name: 'Analytics', icon: <AnalyticsIcon sx={{ fontSize: 18 }} />, href: '/analytics' },
    { name: 'Data Explorer', icon: <DataObjectIcon sx={{ fontSize: 18 }} />, href: '/explorer' },
    { name: 'AI Insights', icon: <PsychologyIcon sx={{ fontSize: 18 }} />, href: '/ai-insights' },
    { name: 'Upload Data', icon: <CloudUploadIcon sx={{ fontSize: 18 }} />, href: '/upload' },
    { name: 'Profile', icon: <PersonIcon sx={{ fontSize: 18 }} />, href: '/profile' },
  ];

  const platformFeatures = [
    { 
      name: 'Real-time Analytics', 
      icon: <TrendingUpIcon />, 
      description: 'Live aquatic data monitoring',
      gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    },
    { 
      name: 'AI-Powered Insights', 
      icon: <PsychologyIcon />, 
      description: 'Machine learning analysis',
      gradient: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)'
    },
    { 
      name: 'Custom Dashboards', 
      icon: <DashboardIcon />, 
      description: 'Personalized views',
      gradient: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)'
    },
  ];

  const aquaticParameters = [
    { name: 'Temperature', icon: <ThermostatIcon sx={{ fontSize: 16 }} />, unit: '°C' },
    { name: 'Salinity', icon: <OpacityIcon sx={{ fontSize: 16 }} />, unit: 'PSU' },
    { name: 'pH Levels', icon: <NatureIcon sx={{ fontSize: 16 }} />, unit: 'pH' },
    { name: 'Depth', icon: <TrendingUpIcon sx={{ fontSize: 16 }} />, unit: 'm' },
  ];

  const resources = [
    { label: 'Documentation', href: '/docs', icon: <DescriptionIcon sx={{ fontSize: 18 }} /> },
    { label: 'API Reference', href: '/api-docs', icon: <CodeIcon sx={{ fontSize: 18 }} /> },
    { label: 'Tutorials', href: '/tutorials', icon: <SchoolIcon sx={{ fontSize: 18 }} /> },
    { label: 'Blog', href: '/blog', icon: <DescriptionIcon sx={{ fontSize: 18 }} /> },
  ];

  const legalLinks = [
    { label: 'Privacy Policy', href: '#privacy' },
    { label: 'Terms of Service', href: '#terms' },
    { label: 'Cookie Policy', href: '#cookies' },
    { label: 'Compliance', href: '#compliance' },
  ];

  const socialLinks = [
    { icon: <TwitterIcon />, href: 'https://twitter.com', label: 'Twitter', color: '#1DA1F2' },
    { icon: <LinkedInIcon />, href: 'https://linkedin.com', label: 'LinkedIn', color: '#0A66C2' },
    { icon: <GitHubIcon />, href: 'https://github.com', label: 'GitHub', color: '#333' },
    { icon: <YouTubeIcon />, href: 'https://youtube.com', label: 'YouTube', color: '#FF0000' },
    { icon: <EmailIcon />, href: 'mailto:contact@meenasetu.ai', label: 'Email', color: '#EA4335' },
  ];

  return (
    <Box
      component="footer"
      sx={{
        mt: 'auto',
        position: 'relative',
        background: 'linear-gradient(135deg, #0277bd 0%, #00897b 100%)',
        color: 'white',
        overflow: 'hidden',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '4px',
          background: 'linear-gradient(90deg, #4ecdc4 0%, #45b7d1 50%, #4ecdc4 100%)',
          backgroundSize: '200% 100%',
          animation: 'gradient 3s ease infinite',
        },
        '@keyframes gradient': {
          '0%, 100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
        },
        '&::after': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'radial-gradient(circle at 20% 50%, rgba(78, 205, 196, 0.1) 0%, transparent 50%), radial-gradient(circle at 80% 80%, rgba(69, 183, 209, 0.1) 0%, transparent 50%)',
          pointerEvents: 'none',
        },
      }}
    >
      <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1 }}>
        {/* Main Footer Content */}
        <Box sx={{ py: { xs: 6, md: 8 } }}>
          <Grid container spacing={{ xs: 4, md: 4 }}>
            {/* Brand Section */}
            <Grid item xs={12} md={3}>
              <Box sx={{ mb: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                  <Box
                    sx={{
                      width: 56,
                      height: 56,
                      borderRadius: 3,
                      background: 'linear-gradient(135deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.1) 100%)',
                      backdropFilter: 'blur(10px)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      boxShadow: '0 8px 24px rgba(0,0,0,0.2)',
                      border: '1px solid rgba(255,255,255,0.2)',
                      transition: 'transform 0.3s ease',
                      cursor: 'pointer',
                      '&:hover': {
                        transform: 'scale(1.05) rotate(5deg)',
                      },
                    }}
                  >
                    <WavesIcon sx={{ fontSize: 32, color: 'white' }} />
                  </Box>
                  <Box>
                    <Typography
                      variant="h6"
                      sx={{
                        fontWeight: 800,
                        letterSpacing: '-0.5px',
                        color: 'white',
                      }}
                    >
                      MeenaSetu AI
                    </Typography>
                    <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.8)', fontWeight: 500 }}>
                      Aquatic Intelligence Platform
                    </Typography>
                  </Box>
                </Box>

                <Typography
                  variant="body2"
                  sx={{ 
                    color: 'rgba(255,255,255,0.9)', 
                    mb: 3, 
                    lineHeight: 1.7,
                  }}
                >
                  Advanced aquatic data platform powered by AI. Monitor, analyze, and understand our waters like never before.
                </Typography>

                {/* Contact Info */}
                <Stack spacing={1} sx={{ mb: 3 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <EmailIcon sx={{ fontSize: 16, color: 'rgba(255,255,255,0.8)' }} />
                    <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.9)' }}>
                      contact@meenasetu.ai
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <PhoneIcon sx={{ fontSize: 16, color: 'rgba(255,255,255,0.8)' }} />
                    <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.9)' }}>
                      +91 620-667-8489
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <LocationOnIcon sx={{ fontSize: 16, color: 'rgba(255,255,255,0.8)' }} />
                    <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.9)' }}>
                      Patna, Bihar, India
                    </Typography>
                  </Box>
                </Stack>

                {/* Social Links */}
                <Stack direction="row" spacing={1} sx={{ flexWrap: 'wrap' }}>
                  {socialLinks.map((social) => (
                    <IconButton
                      key={social.label}
                      href={social.href}
                      target="_blank"
                      rel="noopener noreferrer"
                      aria-label={social.label}
                      sx={{
                        width: 40,
                        height: 40,
                        border: '1px solid',
                        borderColor: 'rgba(255,255,255,0.3)',
                        color: 'white',
                        transition: 'all 0.3s ease',
                        '&:hover': {
                          borderColor: 'white',
                          backgroundColor: 'rgba(255,255,255,0.2)',
                          transform: 'translateY(-4px)',
                          boxShadow: `0 8px 16px rgba(0,0,0,0.3)`,
                        },
                      }}
                    >
                      {social.icon}
                    </IconButton>
                  ))}
                </Stack>
              </Box>
            </Grid>

            {/* Quick Links */}
            <Grid item xs={12} sm={6} md={3}>
              <Typography
                variant="subtitle2"
                sx={{
                  fontWeight: 700,
                  mb: 2,
                  color: 'white',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                  fontSize: '0.75rem',
                }}
              >
                Quick Links
              </Typography>
              <Stack spacing={1.5}>
                {quickLinks.map((link) => (
                  <Link
                    key={link.name}
                    href={link.href}
                    underline="none"
                    sx={{
                      color: 'rgba(255,255,255,0.9)',
                      fontSize: '0.875rem',
                      fontWeight: 500,
                      transition: 'all 0.2s ease',
                      position: 'relative',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 1,
                      width: 'fit-content',
                      '&:hover': {
                        color: 'white',
                        transform: 'translateX(4px)',
                        '& .arrow-icon': {
                          opacity: 1,
                          transform: 'translateX(0)',
                        }
                      },
                    }}
                  >
                    {link.icon}
                    {link.name}
                    <ArrowForwardIcon 
                      className="arrow-icon"
                      sx={{ 
                        fontSize: 14, 
                        opacity: 0,
                        transform: 'translateX(-4px)',
                        transition: 'all 0.2s ease'
                      }} 
                    />
                  </Link>
                ))}
              </Stack>
            </Grid>

            {/* Resources */}
            <Grid item xs={12} sm={6} md={3}>
              <Typography
                variant="subtitle2"
                sx={{
                  fontWeight: 700,
                  mb: 2,
                  color: 'white',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                  fontSize: '0.75rem',
                }}
              >
                Resources
              </Typography>
              <Stack spacing={1.5}>
                {resources.map((link) => (
                  <Link
                    key={link.label}
                    href={link.href}
                    underline="none"
                    sx={{
                      color: 'rgba(255,255,255,0.9)',
                      fontSize: '0.875rem',
                      fontWeight: 500,
                      transition: 'all 0.2s ease',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 1,
                      width: 'fit-content',
                      '&:hover': {
                        color: 'white',
                        transform: 'translateX(4px)',
                      },
                    }}
                  >
                    {link.icon}
                    {link.label}
                  </Link>
                ))}
              </Stack>

              <Typography
                variant="subtitle2"
                sx={{
                  fontWeight: 700,
                  mt: 3,
                  mb: 2,
                  color: 'white',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                  fontSize: '0.75rem',
                }}
              >
                Parameters
              </Typography>
              <Stack spacing={1}>
                {aquaticParameters.map((param) => (
                  <Box key={param.name} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box sx={{ color: '#4ecdc4' }}>{param.icon}</Box>
                    <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.9)', fontSize: '0.875rem' }}>
                      {param.name} ({param.unit})
                    </Typography>
                  </Box>
                ))}
              </Stack>
            </Grid>

            {/* Newsletter */}
            <Grid item xs={12} md={3}>
              <Typography
                variant="subtitle2"
                sx={{
                  fontWeight: 700,
                  mb: 2,
                  color: 'white',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                  fontSize: '0.75rem',
                }}
              >
                Stay Updated
              </Typography>
              <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.9)', mb: 2, lineHeight: 1.6 }}>
                Subscribe to our newsletter for the latest aquatic data insights and platform updates.
              </Typography>
              <Box
                component="form"
                onSubmit={handleNewsletterSubmit}
                sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}
              >
                <TextField
                  size="small"
                  placeholder="Enter your email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      bgcolor: 'rgba(255,255,255,0.15)',
                      backdropFilter: 'blur(10px)',
                      color: 'white',
                      '& fieldset': { borderColor: 'rgba(255,255,255,0.3)' },
                      '&:hover fieldset': { borderColor: 'rgba(255,255,255,0.5)' },
                      '&.Mui-focused fieldset': { borderColor: 'white' },
                    },
                    '& .MuiInputBase-input::placeholder': { color: 'rgba(255,255,255,0.7)' },
                  }}
                />
                <Button
                  type="submit"
                  variant="contained"
                  endIcon={<SendIcon />}
                  sx={{
                    bgcolor: 'rgba(255,255,255,0.2)',
                    backdropFilter: 'blur(10px)',
                    color: 'white',
                    border: '1px solid rgba(255,255,255,0.3)',
                    '&:hover': { 
                      bgcolor: 'rgba(255,255,255,0.3)',
                      transform: 'translateY(-2px)',
                      boxShadow: '0 8px 16px rgba(0,0,0,0.2)',
                    },
                    transition: 'all 0.3s ease',
                  }}
                >
                  Subscribe
                </Button>
              </Box>
            </Grid>
          </Grid>

          {/* Platform Features Bar */}
          <Box
            sx={{
              mt: { xs: 4, md: 6 },
              p: { xs: 2.5, md: 3 },
              borderRadius: 3,
              background: 'rgba(255,255,255,0.1)',
              border: '1px solid rgba(255,255,255,0.2)',
              backdropFilter: 'blur(10px)',
            }}
          >
            <Grid container spacing={2} alignItems="center">
              {platformFeatures.map((feature, index) => (
                <Grid item xs={12} sm={4} key={index}>
                  <Box
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 1.5,
                      justifyContent: { xs: 'center', sm: 'flex-start' },
                      transition: 'transform 0.2s ease',
                      '&:hover': {
                        transform: 'translateX(4px)',
                      }
                    }}
                  >
                    <Box
                      sx={{
                        width: 40,
                        height: 40,
                        borderRadius: 2,
                        background: 'rgba(255,255,255,0.2)',
                        backdropFilter: 'blur(10px)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        border: '1px solid rgba(255,255,255,0.3)',
                        transition: 'transform 0.2s ease',
                        '&:hover': {
                          transform: 'scale(1.1)',
                        },
                        '& svg': { color: 'white', fontSize: 20 },
                      }}
                    >
                      {feature.icon}
                    </Box>
                    <Box>
                      <Typography variant="body2" fontWeight={600} color="white">
                        {feature.name}
                      </Typography>
                      <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.8)' }}>
                        {feature.description}
                      </Typography>
                    </Box>
                  </Box>
                </Grid>
              ))}
            </Grid>
          </Box>
        </Box>

        <Divider sx={{ borderColor: 'rgba(255,255,255,0.2)' }} />

        {/* Bottom Bar */}
        <Box
          sx={{
            py: 3,
            display: 'flex',
            flexDirection: { xs: 'column', md: 'row' },
            justifyContent: 'space-between',
            alignItems: 'center',
            gap: 2,
          }}
        >
          <Typography
            variant="body2"
            sx={{
              color: 'rgba(255,255,255,0.9)',
              fontWeight: 500,
              textAlign: { xs: 'center', md: 'left' },
            }}
          >
            © {currentYear} MeenaSetu AI. All rights reserved. | Protecting our waters through data science.
          </Typography>

          <Stack
            direction={{ xs: 'column', sm: 'row' }}
            spacing={{ xs: 1, sm: 3 }}
            alignItems="center"
            sx={{ flexWrap: 'wrap', justifyContent: 'center' }}
          >
            {legalLinks.map((link, index) => (
              <Link
                key={link.label}
                href={link.href}
                sx={{
                  color: 'rgba(255,255,255,0.8)',
                  textDecoration: 'none',
                  fontSize: '0.875rem',
                  '&:hover': { color: 'white', textDecoration: 'underline' },
                }}
              >
                {link.label}
              </Link>
            ))}
          </Stack>
        </Box>

        {/* Status Indicator */}
        <Box sx={{ display: 'flex', justifyContent: 'center', pb: 2 }}>
          <Chip
            icon={<FiberManualRecordIcon sx={{ fontSize: 12, color: '#10b981 !important' }} />}
            label="All Systems Operational • v2.5.1 • Powered by AI & ML"
            size="small"
            sx={{
              backgroundColor: 'rgba(255,255,255,0.15)',
              backdropFilter: 'blur(10px)',
              color: 'white',
              fontWeight: 600,
              fontSize: '0.7rem',
              height: 28,
              border: '1px solid rgba(255,255,255,0.2)',
              '& .MuiChip-icon': {
                animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
              },
              '@keyframes pulse': {
                '0%, 100%': { opacity: 1 },
                '50%': { opacity: 0.5 },
              },
            }}
          />
        </Box>
      </Container>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={3000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        message={snackbar.message}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        sx={{
          '& .MuiSnackbarContent-root': {
            bgcolor: 'rgba(255,255,255,0.9)',
            color: '#0277bd',
            fontWeight: 600,
          }
        }}
      />
    </Box>
  );
};

export default Footer;