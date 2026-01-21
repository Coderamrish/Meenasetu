import React, { useState, useEffect, useRef } from 'react';
import { 
  Container, 
  Typography, 
  Box, 
  Grid, 
  Card, 
  CardContent,
  Avatar,
  Chip,
  Stack,
  Paper,
  alpha,
  Button,
  IconButton,
  Dialog,
  DialogContent,
  Fab,
  Tooltip,
  useTheme
} from '@mui/material';
import {
  Psychology as PsychologyIcon,
  Storage as StorageIcon,
  Cloud as CloudIcon,
  AutoAwesome as AutoAwesomeIcon,
  Waves as WavesIcon,
  RocketLaunch as RocketLaunchIcon,
  EmojiEvents as EmojiEventsIcon,
  Groups as GroupsIcon,
  Lightbulb as LightbulbIcon,
  LinkedIn as LinkedInIcon,
  GitHub as GitHubIcon,
  Email as EmailIcon,
  Verified as VerifiedIcon,
  ArrowUpward as ArrowUpwardIcon,
  PlayCircle as PlayCircleIcon,
  Star as StarIcon,
  TrendingUp as TrendingUpIcon,
  Speed as SpeedIcon,
  Security as SecurityIcon,
  Share as ShareIcon,
  Close as CloseIcon,
  Mouse as MouseIcon
} from '@mui/icons-material';
import { motion } from 'framer-motion';

// Helper function to extract color from gradient
const extractColorFromGradient = (gradient) => {
  // Extract first color from gradient string
  const match = gradient.match(/#([0-9A-Fa-f]{6}|[0-9A-Fa-f]{3})/);
  return match ? match[0] : '#0277bd';
};

const About = () => {
  const [scrollProgress, setScrollProgress] = useState(0);
  const [stats, setStats] = useState({ users: 0, predictions: 0, accuracy: 0, papers: 0 });
  const [videoOpen, setVideoOpen] = useState(false);
  const containerRef = useRef(null);
  const theme = useTheme();

  // Animation variants
  const fadeInUp = {
    hidden: { opacity: 0, y: 60 },
    visible: { opacity: 1, y: 0 }
  };

  // Scroll progress handler
  useEffect(() => {
    const handleScroll = () => {
      const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
      const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      setScrollProgress((winScroll / height) * 100);
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

  const techStack = [
    { name: 'React.js', level: 95, color: '#61dafb', icon: '⚛️' },
    { name: 'Node.js', level: 92, color: '#68a063', icon: '🟢' },
    { name: 'Python', level: 98, color: '#3776ab', icon: '🐍' },
    { name: 'TensorFlow', level: 90, color: '#ff6f00', icon: '🧠' },
    { name: 'PyTorch', level: 88, color: '#ee4c2c', icon: '🔥' },
    { name: 'MongoDB', level: 85, color: '#47a248', icon: '🍃' },
    { name: 'PostgreSQL', level: 87, color: '#336791', icon: '🐘' },
    { name: 'AWS', level: 82, color: '#ff9900', icon: '☁️' },
  ];

  const features = [
    {
      icon: <PsychologyIcon sx={{ fontSize: 40 }} />,
      title: 'Advanced AI Models',
      description: 'State-of-the-art deep learning models for fish classification and disease detection',
      stats: '98.5% accuracy',
      gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      primaryColor: '#667eea',
      delay: 0
    },
    {
      icon: <WavesIcon sx={{ fontSize: 40 }} />,
      title: 'Computer Vision',
      description: 'Cutting-edge image recognition technology trained on thousands of aquatic species',
      stats: '10,000+ images',
      gradient: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
      primaryColor: '#f093fb',
      delay: 0.1
    },
    {
      icon: <StorageIcon sx={{ fontSize: 40 }} />,
      title: 'Big Data Analytics',
      description: 'Scalable data processing pipeline handling millions of data points',
      stats: '1M+ data points',
      gradient: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
      primaryColor: '#4facfe',
      delay: 0.2
    },
    {
      icon: <CloudIcon sx={{ fontSize: 40 }} />,
      title: 'Cloud Infrastructure',
      description: 'Robust AWS-powered architecture ensuring high availability',
      stats: '99.9% uptime',
      gradient: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
      primaryColor: '#43e97b',
      delay: 0.3
    },
    {
      icon: <AutoAwesomeIcon sx={{ fontSize: 40 }} />,
      title: 'Real-time Processing',
      description: 'Instant AI predictions and analytics with minimal latency',
      stats: '< 2 seconds',
      gradient: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
      primaryColor: '#fa709a',
      delay: 0.4
    },
    {
      icon: <RocketLaunchIcon sx={{ fontSize: 40 }} />,
      title: 'Continuous Innovation',
      description: 'Regular updates with new features based on latest research',
      stats: 'Monthly updates',
      gradient: 'linear-gradient(135deg, #30cfd0 0%, #330867 100%)',
      primaryColor: '#30cfd0',
      delay: 0.5
    },
  ];

  const milestones = [
    { 
      year: '2024', 
      title: 'Platform Launch', 
      description: 'MeenaSetu AI officially launched with core features',
      icon: '🚀',
      color: '#0277bd'
    },
    { 
      year: '2024', 
      title: '50,000+ Predictions', 
      description: 'Crossed major milestone in AI predictions',
      icon: '📈',
      color: '#00897b'
    },
    { 
      year: '2024', 
      title: '98.5% Accuracy', 
      description: 'Achieved industry-leading accuracy rate',
      icon: '🎯',
      color: '#ff6f00'
    },
    { 
      year: '2025', 
      title: 'Global Expansion', 
      description: 'Serving researchers in 50+ countries',
      icon: '🌍',
      color: '#7b1fa2'
    },
  ];

  const achievements = [
    { 
      icon: <EmojiEventsIcon />, 
      text: 'Best AI Innovation 2024', 
      color: '#fbbf24',
      count: '1st'
    },
    { 
      icon: <VerifiedIcon />, 
      text: 'ISO 27001 Certified', 
      color: '#10b981',
      count: '99.9%'
    },
    { 
      icon: <GroupsIcon />, 
      text: 'Active Users', 
      color: '#3b82f6',
      count: `${stats.users}+`
    },
    { 
      icon: <LightbulbIcon />, 
      text: 'Research Papers', 
      color: '#8b5cf6',
      count: `${stats.papers}+`
    },
  ];

  const testimonials = [
    {
      name: 'Dr. Sarah Chen',
      role: 'Marine Biologist',
      content: 'MeenaSetu AI has revolutionized our research. The accuracy is phenomenal!',
      avatar: 'SC'
    },
    {
      name: 'Prof. Rajesh Kumar',
      role: 'Aquaculture Expert',
      content: 'An indispensable tool for modern fish farming. Highly recommended!',
      avatar: 'RK'
    },
    {
      name: 'Lisa Thompson',
      role: 'Conservationist',
      content: 'The platform has helped us identify endangered species more effectively.',
      avatar: 'LT'
    }
  ];

  const teamMembers = [
    {
      name: 'Amrish Tiwary',
      role: 'Founder & Lead Developer',
      avatar: 'AT',
      skills: ['AI/ML', 'Full Stack', 'Cloud']
    },
    {
      name: 'Dr. Meena Sharma',
      role: 'Marine Biology Advisor',
      avatar: 'MS',
      skills: ['Marine Biology', 'Research', 'Conservation']
    },
    {
      name: 'Alex Johnson',
      role: 'AI Research Lead',
      avatar: 'AJ',
      skills: ['Deep Learning', 'CV', 'NLP']
    }
  ];

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <Box 
      ref={containerRef}
      sx={{ 
        bgcolor: 'background.default', 
        minHeight: '100vh',
        position: 'relative',
        overflow: 'hidden',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '100vh',
          background: 'radial-gradient(ellipse at 50% 0%, rgba(2, 119, 189, 0.1) 0%, transparent 60%)',
          pointerEvents: 'none',
          zIndex: 0,
        },
        '&::after': {
          content: '""',
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: '50vh',
          background: 'radial-gradient(ellipse at 50% 100%, rgba(0, 137, 123, 0.1) 0%, transparent 60%)',
          pointerEvents: 'none',
          zIndex: 0,
        }
      }}
    >
      {/* Scroll Progress Bar */}
      <Box
        sx={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          height: 4,
          bgcolor: alpha(theme.palette.primary.main, 0.1),
          zIndex: 9999,
        }}
      >
        <Box
          sx={{
            height: '100%',
            width: `${scrollProgress}%`,
            background: 'linear-gradient(90deg, #0277bd, #00897b)',
            transition: 'width 0.1s ease',
          }}
        />
      </Box>

      {/* Floating Action Buttons */}
      <Box sx={{ position: 'fixed', bottom: 32, right: 32, zIndex: 1000, display: 'flex', flexDirection: 'column', gap: 2 }}>
        <Tooltip title="Scroll to top">
          <Fab
            color="primary"
            size="medium"
            onClick={scrollToTop}
            sx={{
              background: 'linear-gradient(135deg, #0277bd 0%, #00897b 100%)',
              boxShadow: '0 8px 32px rgba(2, 119, 189, 0.3)',
            }}
          >
            <ArrowUpwardIcon />
          </Fab>
        </Tooltip>
        
        <Tooltip title="Share">
          <Fab
            size="medium"
            sx={{
              bgcolor: 'background.paper',
              boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
            }}
          >
            <ShareIcon />
          </Fab>
        </Tooltip>
      </Box>

      {/* Animated Background Elements */}
      <Box
        sx={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          pointerEvents: 'none',
          zIndex: -1,
          overflow: 'hidden',
        }}
      >
        {[...Array(20)].map((_, i) => (
          <Box
            key={i}
            sx={{
              position: 'absolute',
              width: Math.random() * 100 + 50,
              height: Math.random() * 100 + 50,
              background: `radial-gradient(circle, ${alpha('#0277bd', 0.05)} 0%, transparent 70%)`,
              borderRadius: '50%',
              top: `${Math.random() * 100}%`,
              left: `${Math.random() * 100}%`,
              animation: `float ${Math.random() * 20 + 20}s linear infinite`,
              animationDelay: `${Math.random() * 5}s`,
              '@keyframes float': {
                '0%, 100%': { transform: 'translateY(0) rotate(0deg)' },
                '50%': { transform: `translateY(${Math.random() * 100 - 50}px) rotate(${Math.random() * 180 - 90}deg)` },
              }
            }}
          />
        ))}
      </Box>

      {/* Hero Section */}
      <Box
        component={motion.div}
        initial="hidden"
        animate="visible"
        variants={fadeInUp}
        sx={{
          position: 'relative',
          overflow: 'hidden',
          background: 'linear-gradient(135deg, #0a1929 0%, #0d2b36 100%)',
          color: 'white',
          pt: { xs: 12, md: 16 },
          pb: { xs: 8, md: 12 },
          mb: 8,
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: `
              radial-gradient(circle at 20% 80%, rgba(2, 119, 189, 0.15) 0%, transparent 40%),
              radial-gradient(circle at 80% 20%, rgba(0, 137, 123, 0.15) 0%, transparent 40%)
            `,
          },
        }}
      >
        <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1 }}>
          <Box sx={{ textAlign: 'center', mb: 6 }}>
            <motion.div
              initial={{ scale: 0, rotate: -180 }}
              animate={{ scale: 1, rotate: 0 }}
              transition={{ duration: 0.8, type: 'spring' }}
            >
              <Chip
                icon={<AutoAwesomeIcon />}
                label="Pioneering AI in Aquatic Research"
                sx={{
                  mb: 4,
                  backgroundColor: alpha('#fff', 0.1),
                  color: 'white',
                  fontWeight: 700,
                  fontSize: '1rem',
                  py: 2,
                  px: 3,
                  backdropFilter: 'blur(10px)',
                  border: '1px solid rgba(255,255,255,0.2)',
                  animation: 'pulse 2s infinite',
                  '@keyframes pulse': {
                    '0%, 100%': { boxShadow: '0 0 0 0 rgba(255,255,255,0.4)' },
                    '50%': { boxShadow: '0 0 0 10px rgba(255,255,255,0)' },
                  }
                }}
              />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <Typography
                variant="h1"
                component="h1"
                gutterBottom
                sx={{
                  fontWeight: 900,
                  fontSize: { xs: '2.5rem', md: '4rem' },
                  mb: 3,
                  background: 'linear-gradient(135deg, #fff 0%, #4fc3f7 100%)',
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  lineHeight: 1.2,
                }}
              >
                Revolutionizing Aquatic Intelligence
              </Typography>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <Typography
                variant="h5"
                sx={{
                  opacity: 0.9,
                  maxWidth: 800,
                  mx: 'auto',
                  fontWeight: 300,
                  lineHeight: 1.6,
                  mb: 6,
                }}
              >
                Powered by cutting-edge artificial intelligence, MeenaSetu transforms how researchers 
                study and protect marine ecosystems. Join the future of aquatic conservation today.
              </Typography>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.6 }}
            >
              <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} justifyContent="center">
                <Button
                  variant="contained"
                  size="large"
                  startIcon={<PlayCircleIcon />}
                  onClick={() => setVideoOpen(true)}
                  sx={{
                    background: 'linear-gradient(135deg, #0277bd 0%, #00897b 100%)',
                    px: 4,
                    py: 1.5,
                    fontSize: '1.1rem',
                    fontWeight: 700,
                    '&:hover': {
                      transform: 'translateY(-2px)',
                      boxShadow: '0 12px 40px rgba(2, 119, 189, 0.4)',
                    },
                    transition: 'all 0.3s ease',
                  }}
                >
                  Watch Demo
                </Button>
                
                <Button
                  variant="outlined"
                  size="large"
                  sx={{
                    borderColor: alpha('#fff', 0.3),
                    color: 'white',
                    px: 4,
                    py: 1.5,
                    fontSize: '1.1rem',
                    fontWeight: 700,
                    '&:hover': {
                      borderColor: 'white',
                      backgroundColor: alpha('#fff', 0.1),
                      transform: 'translateY(-2px)',
                    },
                    transition: 'all 0.3s ease',
                  }}
                >
                  Get Started
                </Button>
              </Stack>
            </motion.div>
          </Box>

          {/* Live Stats Bar */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
          >
            <Grid container spacing={3} sx={{ mt: 8 }}>
              {achievements.map((achievement, index) => (
                <Grid item xs={6} md={3} key={index}>
                  <motion.div
                    whileHover={{ scale: 1.05, y: -5 }}
                    transition={{ type: 'spring', stiffness: 300 }}
                  >
                    <Paper
                      sx={{
                        p: 3,
                        textAlign: 'center',
                        background: alpha('#fff', 0.05),
                        backdropFilter: 'blur(20px)',
                        border: '1px solid rgba(255,255,255,0.1)',
                        color: 'white',
                        borderRadius: 4,
                        height: '100%',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        position: 'relative',
                        overflow: 'hidden',
                        '&::before': {
                          content: '""',
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          right: 0,
                          height: 4,
                          background: achievement.color,
                        }
                      }}
                    >
                      <Box sx={{ 
                        color: achievement.color, 
                        mb: 2,
                        fontSize: '2.5rem',
                        fontWeight: 700,
                        lineHeight: 1,
                      }}>
                        {achievement.count}
                      </Box>
                      <Box sx={{ color: achievement.color, mb: 1 }}>
                        {achievement.icon}
                      </Box>
                      <Typography variant="body2" fontWeight={600} sx={{ opacity: 0.9 }}>
                        {achievement.text}
                      </Typography>
                    </Paper>
                  </motion.div>
                </Grid>
              ))}
            </Grid>
          </motion.div>
        </Container>

        {/* Scroll Indicator */}
        <Box
          sx={{
            position: 'absolute',
            bottom: 40,
            left: '50%',
            transform: 'translateX(-50%)',
            textAlign: 'center',
            animation: 'bounce 2s infinite',
            '@keyframes bounce': {
              '0%, 100%': { transform: 'translateX(-50%) translateY(0)' },
              '50%': { transform: 'translateX(-50%) translateY(-10px)' },
            }
          }}
        >
          <MouseIcon sx={{ color: alpha('#fff', 0.5), fontSize: 32 }} />
          <Typography variant="caption" sx={{ display: 'block', color: alpha('#fff', 0.5), mt: 1 }}>
            Scroll to explore
          </Typography>
        </Box>
      </Box>

      <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1 }}>
        {/* Mission & Vision - Animated */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
        >
          <Grid container spacing={4} sx={{ mb: 10 }}>
            <Grid item xs={12} md={6}>
              <motion.div
                whileHover={{ scale: 1.02 }}
                transition={{ type: 'spring', stiffness: 300 }}
              >
                <Card
                  sx={{
                    height: '100%',
                    background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)',
                    backdropFilter: 'blur(10px)',
                    border: '1px solid rgba(102, 126, 234, 0.2)',
                    position: 'relative',
                    overflow: 'hidden',
                    '&::before': {
                      content: '""',
                      position: 'absolute',
                      top: -50,
                      right: -50,
                      width: 100,
                      height: 100,
                      borderRadius: '50%',
                      background: 'radial-gradient(circle, rgba(102, 126, 234, 0.2) 0%, transparent 70%)',
                    }
                  }}
                >
                  <CardContent sx={{ p: 5, position: 'relative', zIndex: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                      <Box sx={{
                        width: 48,
                        height: 48,
                        borderRadius: 2,
                        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        mr: 2,
                        color: 'white',
                      }}>
                        <EmojiEventsIcon />
                      </Box>
                      <Typography variant="h4" fontWeight={800}>
                        Our Mission
                      </Typography>
                    </Box>
                    <Typography variant="body1" sx={{ lineHeight: 1.8, fontSize: '1.1rem' }}>
                      To democratize access to advanced AI technology for aquatic research, 
                      making world-class fish species identification and disease detection 
                      accessible to researchers, conservationists, and aquaculture professionals 
                      worldwide. We're committed to protecting marine biodiversity through 
                      innovative technology that bridges the gap between cutting-edge AI and 
                      practical conservation efforts.
                    </Typography>
                  </CardContent>
                </Card>
              </motion.div>
            </Grid>

            <Grid item xs={12} md={6}>
              <motion.div
                whileHover={{ scale: 1.02 }}
                transition={{ type: 'spring', stiffness: 300, delay: 0.1 }}
              >
                <Card
                  sx={{
                    height: '100%',
                    background: 'linear-gradient(135deg, rgba(240, 147, 251, 0.1) 0%, rgba(245, 87, 108, 0.1) 100%)',
                    backdropFilter: 'blur(10px)',
                    border: '1px solid rgba(240, 147, 251, 0.2)',
                    position: 'relative',
                    overflow: 'hidden',
                    '&::before': {
                      content: '""',
                      position: 'absolute',
                      top: -50,
                      right: -50,
                      width: 100,
                      height: 100,
                      borderRadius: '50%',
                      background: 'radial-gradient(circle, rgba(240, 147, 251, 0.2) 0%, transparent 70%)',
                    }
                  }}
                >
                  <CardContent sx={{ p: 5, position: 'relative', zIndex: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                      <Box sx={{
                        width: 48,
                        height: 48,
                        borderRadius: 2,
                        background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        mr: 2,
                        color: 'white',
                      }}>
                        <LightbulbIcon />
                      </Box>
                      <Typography variant="h4" fontWeight={800}>
                        Our Vision
                      </Typography>
                    </Box>
                    <Typography variant="body1" sx={{ lineHeight: 1.8, fontSize: '1.1rem' }}>
                      To become the world's leading AI platform for aquatic intelligence, 
                      enabling groundbreaking discoveries in marine biology, sustainable 
                      aquaculture, and ocean conservation. We envision a future where AI-powered 
                      insights help preserve our oceans for generations to come, creating a 
                      sustainable balance between technological advancement and environmental 
                      stewardship.
                    </Typography>
                  </CardContent>
                </Card>
              </motion.div>
            </Grid>
          </Grid>
        </motion.div>

        {/* Platform Features with 3D Effect */}
        <Box sx={{ mb: 12 }}>
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <Typography
              variant="h2"
              textAlign="center"
              gutterBottom
              fontWeight={900}
              sx={{ 
                mb: 2,
                background: 'linear-gradient(135deg, #0277bd 0%, #00897b 100%)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              Advanced Platform Features
            </Typography>
            <Typography
              variant="h6"
              textAlign="center"
              color="text.secondary"
              sx={{ mb: 8, maxWidth: 800, mx: 'auto' }}
            >
              Built with cutting-edge technology to deliver exceptional performance, accuracy, and user experience
            </Typography>
          </motion.div>

          <Grid container spacing={4}>
            {features.map((feature, index) => (
              <Grid item xs={12} md={4} key={index}>
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  whileHover={{ 
                    scale: 1.05,
                    rotateX: 5,
                    rotateY: 5,
                    transition: { type: 'spring', stiffness: 300 }
                  }}
                  viewport={{ once: true }}
                  transition={{ delay: feature.delay }}
                  style={{ perspective: 1000 }}
                >
                  <Card
                    sx={{
                      height: '100%',
                      position: 'relative',
                      overflow: 'hidden',
                      transition: 'all 0.3s ease',
                      borderRadius: 4,
                      background: 'linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%)',
                      backdropFilter: 'blur(10px)',
                      border: '1px solid rgba(255,255,255,0.3)',
                      boxShadow: '0 20px 60px rgba(0,0,0,0.1)',
                      '&::before': {
                        content: '""',
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        height: 4,
                        background: feature.gradient,
                      },
                    }}
                  >
                    <CardContent sx={{ p: 4, height: '100%', display: 'flex', flexDirection: 'column' }}>
                      <Box
                        sx={{
                          width: 80,
                          height: 80,
                          borderRadius: 3,
                          background: feature.gradient,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          mb: 3,
                          color: 'white',
                          alignSelf: 'center',
                          boxShadow: `0 10px 30px ${alpha(feature.primaryColor, 0.3)}`,
                        }}
                      >
                        {feature.icon}
                      </Box>
                      <Typography variant="h5" gutterBottom fontWeight={800} textAlign="center">
                        {feature.title}
                      </Typography>
                      <Typography 
                        variant="body1" 
                        color="text.secondary" 
                        sx={{ 
                          lineHeight: 1.7, 
                          mb: 3,
                          flexGrow: 1,
                          textAlign: 'center'
                        }}
                      >
                        {feature.description}
                      </Typography>
                      <Chip
                        label={feature.stats}
                        size="small"
                        sx={{
                          alignSelf: 'center',
                          background: feature.gradient,
                          color: 'white',
                          fontWeight: 700,
                          fontSize: '0.9rem',
                          py: 1,
                        }}
                      />
                    </CardContent>
                  </Card>
                </motion.div>
              </Grid>
            ))}
          </Grid>
        </Box>

        {/* Creator Section - Enhanced */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
        >
          <Card
            sx={{
              mb: 12,
              background: 'linear-gradient(135deg, rgba(15, 32, 39, 0.95) 0%, rgba(32, 58, 67, 0.95) 50%, rgba(44, 83, 100, 0.95) 100%)',
              color: 'white',
              overflow: 'hidden',
              position: 'relative',
              borderRadius: 4,
              border: '1px solid rgba(255,255,255,0.1)',
              boxShadow: '0 30px 90px rgba(15, 32, 39, 0.4)',
            }}
          >
            <Box
              sx={{
                position: 'absolute',
                top: -100,
                right: -100,
                width: 300,
                height: 300,
                borderRadius: '50%',
                background: 'radial-gradient(circle, rgba(2, 119, 189, 0.2) 0%, transparent 70%)',
                animation: 'rotate 20s linear infinite',
                '@keyframes rotate': {
                  '0%': { transform: 'rotate(0deg)' },
                  '100%': { transform: 'rotate(360deg)' },
                }
              }}
            />
            <CardContent sx={{ p: { xs: 4, md: 6 }, position: 'relative', zIndex: 1 }}>
              <Grid container spacing={6} alignItems="center">
                <Grid item xs={12} md={4} sx={{ textAlign: 'center' }}>
                  <motion.div
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <Avatar
                      sx={{
                        width: 200,
                        height: 200,
                        mx: 'auto',
                        mb: 3,
                        border: '4px solid rgba(255,255,255,0.2)',
                        background: 'linear-gradient(135deg, #0277bd 0%, #00897b 100%)',
                        fontSize: '4rem',
                        fontWeight: 800,
                        boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
                      }}
                    >
                      AT
                    </Avatar>
                  </motion.div>
                  
                  <Stack direction="row" spacing={2} justifyContent="center" sx={{ mb: 4 }}>
                    {[
                      { icon: <LinkedInIcon />, color: '#0077b5', label: 'LinkedIn' },
                      { icon: <GitHubIcon />, color: '#333', label: 'GitHub' },
                      { icon: <EmailIcon />, color: '#ea4335', label: 'Email' },
                    ].map((social, index) => (
                      <Tooltip key={index} title={social.label}>
                        <motion.div whileHover={{ scale: 1.2, rotate: 5 }}>
                          <Avatar 
                            sx={{ 
                              bgcolor: social.color, 
                              width: 48, 
                              height: 48, 
                              cursor: 'pointer',
                              '&:hover': {
                                boxShadow: `0 0 0 4px ${alpha(social.color, 0.3)}`,
                              },
                              transition: 'all 0.3s ease',
                            }}
                          >
                            {social.icon}
                          </Avatar>
                        </motion.div>
                      </Tooltip>
                    ))}
                  </Stack>

                  <Chip
                    icon={<VerifiedIcon />}
                    label="Verified Profile"
                    size="small"
                    sx={{
                      backgroundColor: alpha('#10b981', 0.2),
                      color: '#10b981',
                      fontWeight: 600,
                      border: '1px solid rgba(16, 185, 129, 0.3)',
                    }}
                  />
                </Grid>

                <Grid item xs={12} md={8}>
                  <motion.div
                    initial={{ x: 50 }}
                    animate={{ x: 0 }}
                    transition={{ delay: 0.2 }}
                  >
                    <Chip
                      label="Creator & Lead Developer"
                      size="medium"
                      sx={{
                        mb: 3,
                        backgroundColor: alpha('#fff', 0.1),
                        color: 'white',
                        fontWeight: 700,
                        fontSize: '0.9rem',
                        py: 1,
                      }}
                    />
                    <Typography variant="h2" gutterBottom fontWeight={900}>
                      Amrish Kumar Tiwary
                    </Typography>
                    <Typography variant="h5" gutterBottom sx={{ opacity: 0.9, mb: 4, color: '#4fc3f7' }}>
                      Full Stack AI Engineer | Machine Learning Expert | Visionary
                    </Typography>
                    
                    <Box sx={{ mb: 4 }}>
                      <Typography variant="body1" paragraph sx={{ lineHeight: 1.8, opacity: 0.95, mb: 2 }}>
                        A passionate Full Stack AI Engineer with expertise in building intelligent systems 
                        that solve real-world environmental challenges. With extensive experience in machine learning, 
                        computer vision, and full-stack development, Amrish created MeenaSetu AI to bridge 
                        the critical gap between cutting-edge AI technology and aquatic research needs.
                      </Typography>
                      <Typography variant="body1" sx={{ lineHeight: 1.8, opacity: 0.95 }}>
                        Specializing in Python, TensorFlow, PyTorch, React, and cloud technologies, 
                        he leads a team dedicated to leveraging AI for environmental conservation and sustainable 
                        development. His vision for MeenaSetu AI is to create an accessible platform that 
                        empowers researchers and conservationists worldwide with powerful AI tools.
                      </Typography>
                    </Box>

                    <Stack direction="row" spacing={2} sx={{ flexWrap: 'wrap', gap: 1.5, mt: 4 }}>
                      {[
                        { skill: 'AI/ML', level: 98, color: '#0277bd' },
                        { skill: 'Deep Learning', level: 96, color: '#00897b' },
                        { skill: 'Full Stack', level: 95, color: '#ff6f00' },
                        { skill: 'Cloud Computing', level: 92, color: '#7b1fa2' },
                        { skill: 'Computer Vision', level: 94, color: '#d32f2f' },
                      ].map((item, index) => (
                        <motion.div
                          key={index}
                          whileHover={{ scale: 1.05 }}
                        >
                          <Chip
                            label={`${item.skill} ${item.level}%`}
                            sx={{
                              backgroundColor: alpha(item.color, 0.1),
                              color: item.color,
                              fontWeight: 700,
                              fontSize: '0.9rem',
                              py: 1,
                              px: 2,
                              border: `1px solid ${alpha(item.color, 0.3)}`,
                              '& .MuiChip-label': {
                                display: 'flex',
                                alignItems: 'center',
                                gap: 1,
                              }
                            }}
                            icon={<TrendingUpIcon sx={{ fontSize: 16, color: item.color }} />}
                          />
                        </motion.div>
                      ))}
                    </Stack>
                  </motion.div>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </motion.div>

        {/* Tech Stack with Animated Progress */}
        <Box sx={{ mb: 12 }}>
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <Typography variant="h2" textAlign="center" gutterBottom fontWeight={900} sx={{ mb: 2 }}>
              Technology Stack
            </Typography>
            <Typography
              variant="h6"
              textAlign="center"
              color="text.secondary"
              sx={{ mb: 8, maxWidth: 800, mx: 'auto' }}
            >
              Built with industry-leading technologies and frameworks for maximum performance and scalability
            </Typography>
          </motion.div>

          <Grid container spacing={4}>
            {techStack.map((tech, index) => (
              <Grid item xs={12} sm={6} md={3} key={index}>
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  whileHover={{ scale: 1.05 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                >
                  <Card
                    sx={{
                      height: '100%',
                      position: 'relative',
                      overflow: 'hidden',
                      borderRadius: 3,
                      background: 'linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%)',
                      backdropFilter: 'blur(10px)',
                      border: '1px solid rgba(255,255,255,0.3)',
                      '&::before': {
                        content: '""',
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        height: 4,
                        background: tech.color,
                      }
                    }}
                  >
                    <CardContent sx={{ p: 3 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                        <Typography variant="h4" sx={{ mr: 2 }}>
                          {tech.icon}
                        </Typography>
                        <Typography variant="h6" fontWeight={700} sx={{ flexGrow: 1 }}>
                          {tech.name}
                        </Typography>
                        <Typography variant="body1" fontWeight={800} color={tech.color}>
                          {tech.level}%
                        </Typography>
                      </Box>
                      
                      <Box sx={{ position: 'relative', height: 8, bgcolor: alpha(tech.color, 0.1), borderRadius: 4, overflow: 'hidden' }}>
                        <motion.div
                          initial={{ width: 0 }}
                          whileInView={{ width: `${tech.level}%` }}
                          viewport={{ once: true }}
                          transition={{ duration: 1.5, delay: index * 0.2 }}
                          style={{
                            height: '100%',
                            backgroundColor: tech.color,
                            borderRadius: 4,
                          }}
                        />
                      </Box>
                      
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                        Expert proficiency
                      </Typography>
                    </CardContent>
                  </Card>
                </motion.div>
              </Grid>
            ))}
          </Grid>
        </Box>

        {/* Interactive Timeline */}
        <Box sx={{ mb: 12 }}>
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <Typography variant="h2" textAlign="center" gutterBottom fontWeight={900} sx={{ mb: 8 }}>
              Our Journey Timeline
            </Typography>
          </motion.div>

          <Box sx={{ position: 'relative', pl: { xs: 0, md: 8 } }}>
            {/* Vertical line */}
            <Box
              sx={{
                position: 'absolute',
                left: { xs: 32, md: 64 },
                top: 0,
                bottom: 0,
                width: 4,
                background: 'linear-gradient(180deg, #0277bd 0%, #00897b 100%)',
                display: { xs: 'none', md: 'block' },
              }}
            />

            {milestones.map((milestone, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -50 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.2 }}
              >
                <Box sx={{ 
                  position: 'relative', 
                  mb: 6,
                  pl: { xs: 8, md: 12 },
                  '&::before': {
                    content: '""',
                    position: 'absolute',
                    left: { xs: 24, md: 56 },
                    top: 24,
                    width: 20,
                    height: 20,
                    borderRadius: '50%',
                    background: milestone.color,
                    border: '4px solid white',
                    boxShadow: `0 0 0 4px ${alpha(milestone.color, 0.25)}`,
                    zIndex: 2,
                  }
                }}>
                  <Card
                    sx={{
                      borderRadius: 4,
                      overflow: 'hidden',
                      position: 'relative',
                      '&:hover': {
                        transform: 'translateX(10px)',
                        '& .timeline-icon': {
                          transform: 'scale(1.2) rotate(10deg)',
                        }
                      },
                      transition: 'all 0.3s ease',
                    }}
                  >
                    <CardContent sx={{ p: 4 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                        <Box
                          className="timeline-icon"
                          sx={{
                            width: 48,
                            height: 48,
                            borderRadius: 2,
                            background: `linear-gradient(135deg, ${milestone.color} 0%, ${alpha(milestone.color, 0.7)} 100%)`,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            mr: 3,
                            color: 'white',
                            fontSize: '1.5rem',
                            transition: 'transform 0.3s ease',
                          }}
                        >
                          {milestone.icon}
                        </Box>
                        <Box sx={{ flexGrow: 1 }}>
                          <Chip
                            label={milestone.year}
                            size="small"
                            sx={{
                              mb: 1,
                              background: `linear-gradient(135deg, ${milestone.color} 0%, ${alpha(milestone.color, 0.7)} 100%)`,
                              color: 'white',
                              fontWeight: 800,
                            }}
                          />
                          <Typography variant="h5" fontWeight={800}>
                            {milestone.title}
                          </Typography>
                        </Box>
                      </Box>
                      <Typography variant="body1" color="text.secondary">
                        {milestone.description}
                      </Typography>
                    </CardContent>
                  </Card>
                </Box>
              </motion.div>
            ))}
          </Box>
        </Box>

        {/* Testimonials Carousel */}
        <Box sx={{ mb: 12 }}>
          <Typography variant="h2" textAlign="center" gutterBottom fontWeight={900} sx={{ mb: 8 }}>
            What Our Users Say
          </Typography>

          <Grid container spacing={4}>
            {testimonials.map((testimonial, index) => (
              <Grid item xs={12} md={4} key={index}>
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  whileHover={{ scale: 1.05 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                >
                  <Card
                    sx={{
                      height: '100%',
                      background: 'linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%)',
                      backdropFilter: 'blur(10px)',
                      border: '1px solid rgba(255,255,255,0.3)',
                      borderRadius: 4,
                      position: 'relative',
                      overflow: 'hidden',
                      '&::before': {
                        content: '"❝"',
                        position: 'absolute',
                        top: 20,
                        right: 20,
                        fontSize: '4rem',
                        color: alpha('#0277bd', 0.1),
                        fontFamily: 'Georgia, serif',
                      }
                    }}
                  >
                    <CardContent sx={{ p: 4, height: '100%', display: 'flex', flexDirection: 'column' }}>
                      <Typography variant="body1" paragraph sx={{ lineHeight: 1.8, mb: 4, flexGrow: 1, fontStyle: 'italic' }}>
                        "{testimonial.content}"
                      </Typography>
                      
                      <Box sx={{ display: 'flex', alignItems: 'center', mt: 'auto' }}>
                        <Avatar
                          sx={{
                            width: 56,
                            height: 56,
                            mr: 2,
                            background: 'linear-gradient(135deg, #0277bd 0%, #00897b 100%)',
                            fontWeight: 800,
                          }}
                        >
                          {testimonial.avatar}
                        </Avatar>
                        <Box>
                          <Typography variant="h6" fontWeight={800}>
                            {testimonial.name}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {testimonial.role}
                          </Typography>
                          <Stack direction="row" spacing={0.5} sx={{ mt: 0.5 }}>
                            {[...Array(5)].map((_, i) => (
                              <StarIcon key={i} sx={{ fontSize: 16, color: '#fbbf24' }} />
                            ))}
                          </Stack>
                        </Box>
                      </Box>
                    </CardContent>
                  </Card>
                </motion.div>
              </Grid>
            ))}
          </Grid>
        </Box>

        {/* Team Members */}
        <Box sx={{ mb: 12 }}>
          <Typography variant="h2" textAlign="center" gutterBottom fontWeight={900} sx={{ mb: 8 }}>
            Meet Our Team
          </Typography>

          <Grid container spacing={4}>
            {teamMembers.map((member, index) => (
              <Grid item xs={12} md={4} key={index}>
                <motion.div
                  whileHover={{ scale: 1.05, y: -10 }}
                  transition={{ type: 'spring', stiffness: 300 }}
                >
                  <Card
                    sx={{
                      height: '100%',
                      textAlign: 'center',
                      background: 'linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%)',
                      backdropFilter: 'blur(10px)',
                      border: '1px solid rgba(255,255,255,0.3)',
                      borderRadius: 4,
                      overflow: 'hidden',
                      position: 'relative',
                      '&::before': {
                        content: '""',
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        height: 4,
                        background: index === 0 ? '#0277bd' : index === 1 ? '#00897b' : '#ff6f00',
                      }
                    }}
                  >
                    <CardContent sx={{ p: 4 }}>
                      <Avatar
                        sx={{
                          width: 100,
                          height: 100,
                          mx: 'auto',
                          mb: 3,
                          border: '4px solid',
                          borderColor: index === 0 ? '#0277bd' : index === 1 ? '#00897b' : '#ff6f00',
                          background: `linear-gradient(135deg, ${index === 0 ? '#0277bd' : index === 1 ? '#00897b' : '#ff6f00'} 0%, ${alpha(index === 0 ? '#0277bd' : index === 1 ? '#00897b' : '#ff6f00', 0.7)} 100%)`,
                          fontSize: '2.5rem',
                          fontWeight: 800,
                        }}
                      >
                        {member.avatar}
                      </Avatar>
                      
                      <Typography variant="h5" gutterBottom fontWeight={800}>
                        {member.name}
                      </Typography>
                      <Typography variant="body1" color="primary" fontWeight={600} gutterBottom>
                        {member.role}
                      </Typography>
                      
                      <Stack direction="row" spacing={1} justifyContent="center" sx={{ mt: 3, flexWrap: 'wrap', gap: 1 }}>
                        {member.skills.map((skill, skillIndex) => (
                          <Chip
                            key={skillIndex}
                            label={skill}
                            size="small"
                            sx={{
                              backgroundColor: alpha('#0277bd', 0.1),
                              color: '#0277bd',
                              fontWeight: 600,
                              border: '1px solid rgba(2, 119, 189, 0.3)',
                            }}
                          />
                        ))}
                      </Stack>
                    </CardContent>
                  </Card>
                </motion.div>
              </Grid>
            ))}
          </Grid>
        </Box>

        {/* Final CTA */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
        >
          <Card
            sx={{
              p: { xs: 4, md: 8 },
              textAlign: 'center',
              background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)',
              backdropFilter: 'blur(20px)',
              border: '1px solid rgba(102, 126, 234, 0.2)',
              borderRadius: 4,
              position: 'relative',
              overflow: 'hidden',
              '&::before': {
                content: '""',
                position: 'absolute',
                top: -100,
                right: -100,
                width: 300,
                height: 300,
                borderRadius: '50%',
                background: 'radial-gradient(circle, rgba(102, 126, 234, 0.2) 0%, transparent 70%)',
              },
              '&::after': {
                content: '""',
                position: 'absolute',
                bottom: -100,
                left: -100,
                width: 300,
                height: 300,
                borderRadius: '50%',
                background: 'radial-gradient(circle, rgba(118, 75, 162, 0.2) 0%, transparent 70%)',
              }
            }}
          >
            <Box sx={{ position: 'relative', zIndex: 1 }}>
              <Typography variant="h2" gutterBottom fontWeight={900}>
                Join the AI Revolution in Aquatic Research
              </Typography>
              <Typography variant="h5" paragraph sx={{ opacity: 0.9, mb: 6, maxWidth: 800, mx: 'auto' }}>
                Be part of a global community using cutting-edge AI to protect our oceans and advance marine science
              </Typography>
              
              <Stack direction={{ xs: 'column', sm: 'row' }} spacing={3} justifyContent="center" sx={{ mb: 6 }}>
                <Button
                  variant="contained"
                  size="large"
                  startIcon={<RocketLaunchIcon />}
                  sx={{
                    px: 6,
                    py: 2,
                    fontSize: '1.1rem',
                    fontWeight: 800,
                    background: 'linear-gradient(135deg, #0277bd 0%, #00897b 100%)',
                    '&:hover': {
                      transform: 'translateY(-4px)',
                      boxShadow: '0 20px 40px rgba(2, 119, 189, 0.4)',
                    },
                    transition: 'all 0.3s ease',
                  }}
                >
                  Start Free Trial
                </Button>
                
                <Button
                  variant="outlined"
                  size="large"
                  sx={{
                    px: 6,
                    py: 2,
                    fontSize: '1.1rem',
                    fontWeight: 800,
                    borderColor: '#0277bd',
                    color: '#0277bd',
                    '&:hover': {
                      backgroundColor: alpha('#0277bd', 0.1),
                      borderColor: '#0277bd',
                      transform: 'translateY(-4px)',
                    },
                    transition: 'all 0.3s ease',
                  }}
                >
                  Schedule Demo
                </Button>
              </Stack>

              <Stack direction={{ xs: 'column', sm: 'row' }} spacing={3} justifyContent="center">
                <Chip
                  icon={<SpeedIcon />}
                  label={`${stats.users.toLocaleString()}+ Active Researchers`}
                  sx={{
                    backgroundColor: alpha('#0277bd', 0.1),
                    color: '#0277bd',
                    fontWeight: 700,
                    fontSize: '1rem',
                    py: 2,
                    px: 3,
                  }}
                />
                <Chip
                  icon={<SecurityIcon />}
                  label={`${stats.predictions.toLocaleString()}+ Predictions Made`}
                  sx={{
                    backgroundColor: alpha('#00897b', 0.1),
                    color: '#00897b',
                    fontWeight: 700,
                    fontSize: '1rem',
                    py: 2,
                    px: 3,
                  }}
                />
              </Stack>
            </Box>
          </Card>
        </motion.div>
      </Container>

      {/* Video Dialog */}
      <Dialog
        open={videoOpen}
        onClose={() => setVideoOpen(false)}
        maxWidth="md"
        fullWidth
        PaperProps={{
          sx: {
            borderRadius: 4,
            overflow: 'hidden',
            background: 'linear-gradient(135deg, #0a1929 0%, #0d2b36 100%)',
          }
        }}
      >
        <DialogContent sx={{ p: 0, position: 'relative', aspectRatio: '16/9' }}>
          <IconButton
            onClick={() => setVideoOpen(false)}
            sx={{
              position: 'absolute',
              top: 16,
              right: 16,
              bgcolor: 'rgba(0,0,0,0.5)',
              color: 'white',
              zIndex: 2,
              '&:hover': {
                bgcolor: 'rgba(0,0,0,0.7)',
              }
            }}
          >
            <CloseIcon />
          </IconButton>
          
          <Box
            sx={{
              width: '100%',
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'linear-gradient(135deg, #0a1929 0%, #0d2b36 100%)',
            }}
          >
            <Typography color="white" variant="h6">
              Demo Video Placeholder
            </Typography>
          </Box>
        </DialogContent>
      </Dialog>
    </Box>
  );
};

export default About;